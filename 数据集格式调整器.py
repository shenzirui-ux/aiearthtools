import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QMessageBox, QPushButton, QProgressBar,
    QVBoxLayout, QWidget, QTextEdit
)
from PyQt5.QtGui import QFont

# Pillow 版本兼容
try:
    RESAMPLE_METHOD = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE_METHOD = Image.LANCZOS


class WorkerSignals(QObject):
    progress = pyqtSignal(int, int, float)  # current, total, elapsed
    finished = pyqtSignal(int, int)         # success_count, total
    message = pyqtSignal(str)


class ProcessAndSaveTask(QRunnable):
    def __init__(self, file_list, target_folder, width, height, out_format):
        super().__init__()
        self.file_list = file_list
        self.target_folder = target_folder
        self.width = width
        self.height = height
        self.out_format = out_format
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        total = len(self.file_list)
        start_time = time.time()
        count = 0

        def process_file(file_path):
            try:
                with Image.open(file_path) as img:
                    img = img.resize((self.width, self.height), RESAMPLE_METHOD)
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    new_file = base_name + "." + self.out_format
                    out_path = os.path.join(self.target_folder, new_file)
                    img.save(out_path, self.out_format.upper())
                    return True
            except Exception as e:
                self.signals.message.emit(f"处理失败：{file_path}, 错误: {e}")
                return False

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_file, fp): fp for fp in self.file_list}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    count += 1
                done_count = sum(1 for f in futures if f.done())
                elapsed = time.time() - start_time
                self.signals.progress.emit(done_count, total, elapsed)
        self.signals.finished.emit(count, total)


class CompressTask(QRunnable):
    def __init__(self, dataset_folder):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        start = time.time()
        try:
            archive_path = shutil.make_archive(self.dataset_folder, 'zip', root_dir=self.dataset_folder)
            elapsed = time.time() - start
            self.signals.message.emit(f"压缩完成：{archive_path} 耗时 {elapsed:.1f}s")
            self.signals.finished.emit(1, 1)  # 仅用于通知完成
        except Exception as e:
            self.signals.message.emit(f"压缩失败: {e}")


class DatasetFormatAdjuster(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据集格式调整器")
        self.resize(1000, 800)
        self.setStyleSheet(self.load_stylesheet())

        # 全局变量
        self.dataset_folder = ""   # TARGET_EXTRACTION 文件夹路径
        self.mask_files = []       # 选取的 mask 文件列表（全路径）
        self.image_files = []      # 选取的图像文件列表（全路径）

        self.threadpool = QThreadPool()

        # 主界面区域
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 1. 数据集设置区域
        dataset_group = QGroupBox("数据集设置")
        dataset_layout = QHBoxLayout()
        dataset_group.setLayout(dataset_layout)
        self.create_dataset_button = QPushButton("创建数据集文件夹")
        self.create_dataset_button.clicked.connect(self.create_dataset_folder)
        dataset_layout.addWidget(self.create_dataset_button)
        self.compress_button = QPushButton("压缩数据集")
        self.compress_button.clicked.connect(self.compress_dataset)
        dataset_layout.addWidget(self.compress_button)
        self.dataset_label = QLabel("未创建")
        self.dataset_label.setStyleSheet("color: red;")
        dataset_layout.addWidget(self.dataset_label)
        main_layout.addWidget(dataset_group)

        # 2. Mask 文件处理区域
        mask_group = QGroupBox("Mask 文件处理")
        mask_layout = QVBoxLayout()
        mask_group.setLayout(mask_layout)
        top_mask_layout = QHBoxLayout()
        self.select_mask_button = QPushButton("选取mask文件夹")
        self.select_mask_button.clicked.connect(self.select_mask_folder)
        top_mask_layout.addWidget(self.select_mask_button)
        mask_layout.addLayout(top_mask_layout)
        self.mask_listwidget = QListWidget()
        mask_layout.addWidget(self.mask_listwidget)
        param_mask_layout = QHBoxLayout()
        param_mask_layout.addWidget(QLabel("目标宽度:"))
        self.mask_width_entry = QLineEdit("512")
        self.mask_width_entry.setFixedWidth(50)
        param_mask_layout.addWidget(self.mask_width_entry)
        param_mask_layout.addWidget(QLabel("目标高度:"))
        self.mask_height_entry = QLineEdit("512")
        self.mask_height_entry.setFixedWidth(50)
        param_mask_layout.addWidget(self.mask_height_entry)
        param_mask_layout.addWidget(QLabel("输出格式:"))
        self.mask_format_combo = QComboBox()
        self.mask_format_combo.addItems(["png", "jpg", "tif"])
        self.mask_format_combo.setCurrentText("png")
        self.mask_format_combo.setFixedWidth(60)
        param_mask_layout.addWidget(self.mask_format_combo)
        self.save_mask_button = QPushButton("保存mask文件")
        self.save_mask_button.clicked.connect(self.save_mask_files)
        param_mask_layout.addWidget(self.save_mask_button)
        mask_layout.addLayout(param_mask_layout)
        main_layout.addWidget(mask_group)

        # 3. 图像文件处理区域
        image_group = QGroupBox("图像文件处理")
        image_layout = QVBoxLayout()
        image_group.setLayout(image_layout)
        top_image_layout = QHBoxLayout()
        self.select_image_button = QPushButton("选取图像文件夹")
        self.select_image_button.clicked.connect(self.select_image_folder)
        top_image_layout.addWidget(self.select_image_button)
        image_layout.addLayout(top_image_layout)
        self.image_listwidget = QListWidget()
        image_layout.addWidget(self.image_listwidget)
        param_image_layout = QHBoxLayout()
        param_image_layout.addWidget(QLabel("目标宽度:"))
        self.image_width_entry = QLineEdit("512")
        self.image_width_entry.setFixedWidth(50)
        param_image_layout.addWidget(self.image_width_entry)
        param_image_layout.addWidget(QLabel("目标高度:"))
        self.image_height_entry = QLineEdit("512")
        self.image_height_entry.setFixedWidth(50)
        param_image_layout.addWidget(self.image_height_entry)
        param_image_layout.addWidget(QLabel("输出格式:"))
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["png", "jpg", "tif"])
        self.image_format_combo.setCurrentText("png")
        self.image_format_combo.setFixedWidth(60)
        param_image_layout.addWidget(self.image_format_combo)
        self.save_image_button = QPushButton("保存图像文件")
        self.save_image_button.clicked.connect(self.save_image_files)
        param_image_layout.addWidget(self.save_image_button)
        image_layout.addLayout(param_image_layout)
        main_layout.addWidget(image_group)

        # 4. lst.txt 生成区域
        lst_group = QGroupBox("生成lst.txt")
        lst_layout = QHBoxLayout()
        lst_group.setLayout(lst_layout)
        self.generate_lst_button = QPushButton("生成lst.txt")
        self.generate_lst_button.clicked.connect(self.generate_lst)
        lst_layout.addWidget(self.generate_lst_button)
        main_layout.addWidget(lst_group)

        # 5. 状态信息显示区域
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        main_layout.addWidget(self.status_text)

        # 6. 进度条显示区域
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(400)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("0% - 0.0s")
        progress_layout.addWidget(self.progress_label)
        main_layout.addLayout(progress_layout)

        self.append_status("欢迎使用数据集格式调整器")

    def load_stylesheet(self):
        # QSS 样式设置现代暗色调风格
        return """
        QMainWindow { background-color: #2E2E2E; }
        QGroupBox { 
            font-size: 16px; 
            color: #FFFFFF; 
            border: 1px solid #555555; 
            border-radius: 5px; 
            margin-top: 10px; 
        }
        QGroupBox::title { 
            subcontrol-origin: margin; 
            left: 10px; 
            padding: 0 5px; 
        }
        QLabel { color: #FFFFFF; }
        QPushButton { 
            background-color: #4CAF50; 
            color: #FFFFFF; 
            border: none; 
            border-radius: 5px; 
            padding: 8px 15px; 
            font-size: 14px; 
        }
        QPushButton:hover { background-color: #45A049; }
        QLineEdit, QComboBox, QListWidget, QTextEdit, QProgressBar { 
            background-color: #3E3E3E; 
            color: #FFFFFF; 
            border: 1px solid #555555; 
            border-radius: 3px; 
        }
        QProgressBar::chunk { background-color: #4CAF50; }
        """

    def append_status(self, message):
        self.status_text.append(message)

    @pyqtSlot(int, int, float)
    def update_progress(self, current, total, elapsed):
        percent = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{percent}% - {elapsed:.1f}s")

    @pyqtSlot()
    def create_dataset_folder(self):
        base_folder = QFileDialog.getExistingDirectory(self, "选择数据集存储路径")
        if not base_folder:
            return
        self.dataset_folder = os.path.join(base_folder, "TARGET_EXTRACTION")
        try:
            os.makedirs(os.path.join(self.dataset_folder, "annotations"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_folder, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_folder, "lst"), exist_ok=True)
            self.dataset_label.setText(self.dataset_folder)
            self.dataset_label.setStyleSheet("color: green;")
            self.append_status(f"数据集文件夹已创建：{self.dataset_folder}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建文件夹失败: {e}")

    @pyqtSlot()
    def compress_dataset(self):
        if not self.dataset_folder:
            QMessageBox.warning(self, "未创建数据集", "请先创建数据集文件夹！")
            return

        task = CompressTask(self.dataset_folder)
        task.signals.message.connect(self.append_status)
        task.signals.finished.connect(lambda s, t: QMessageBox.information(self, "压缩完成", f"压缩完成：{self.dataset_folder}.zip"))
        self.threadpool.start(task)

    @pyqtSlot()
    def select_mask_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择mask文件夹")
        if folder:
            self.mask_files = []
            self.mask_listwidget.clear()
            for file in os.listdir(folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif")):
                    file_path = os.path.join(folder, file)
                    self.mask_files.append(file_path)
                    try:
                        with Image.open(file_path) as img:
                            info = f"{file} - {img.size[0]}x{img.size[1]}"
                    except Exception:
                        info = file
                    self.mask_listwidget.addItem(info)
            self.append_status(f"选取到 {len(self.mask_files)} 个mask文件")

    @pyqtSlot()
    def select_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            self.image_files = []
            self.image_listwidget.clear()
            for file in os.listdir(folder):
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif")):
                    file_path = os.path.join(folder, file)
                    self.image_files.append(file_path)
                    try:
                        with Image.open(file_path) as img:
                            info = f"{file} - {img.size[0]}x{img.size[1]}"
                    except Exception:
                        info = file
                    self.image_listwidget.addItem(info)
            self.append_status(f"选取到 {len(self.image_files)} 个图像文件")

    def process_and_save(self, file_list, target_folder, width, height, out_format):
        if not self.dataset_folder:
            QMessageBox.warning(self, "未创建数据集", "请先创建数据集文件夹！")
            return

        task = ProcessAndSaveTask(file_list, target_folder, width, height, out_format)
        task.signals.progress.connect(self.update_progress)
        task.signals.message.connect(self.append_status)
        task.signals.finished.connect(lambda success, total: QMessageBox.information(
            self, "保存完成", f"成功保存 {success} 张文件到 {target_folder}"))
        self.threadpool.start(task)

    @pyqtSlot()
    def save_mask_files(self):
        if not self.dataset_folder:
            QMessageBox.warning(self, "未创建数据集", "请先创建数据集文件夹！")
            return
        target = os.path.join(self.dataset_folder, "annotations")
        try:
            width = int(self.mask_width_entry.text())
            height = int(self.mask_height_entry.text())
        except ValueError:
            QMessageBox.critical(self, "错误", "请输入正确的分辨率数值！")
            return
        out_format = self.mask_format_combo.currentText()
        self.process_and_save(self.mask_files, target, width, height, out_format)

    @pyqtSlot()
    def save_image_files(self):
        if not self.dataset_folder:
            QMessageBox.warning(self, "未创建数据集", "请先创建数据集文件夹！")
            return
        target = os.path.join(self.dataset_folder, "images")
        try:
            width = int(self.image_width_entry.text())
            height = int(self.image_height_entry.text())
        except ValueError:
            QMessageBox.critical(self, "错误", "请输入正确的分辨率数值！")
            return
        out_format = self.image_format_combo.currentText()
        self.process_and_save(self.image_files, target, width, height, out_format)

    @pyqtSlot()
    def generate_lst(self):
        if not self.dataset_folder:
            QMessageBox.warning(self, "未创建数据集", "请先创建数据集文件夹！")
            return
        images_folder = os.path.join(self.dataset_folder, "images")
        annotations_folder = os.path.join(self.dataset_folder, "annotations")
        lst_folder = os.path.join(self.dataset_folder, "lst")
        os.makedirs(lst_folder, exist_ok=True)
        lst_file = os.path.join(lst_folder, "lst.txt")
        try:
            with open(lst_file, 'w') as f:
                image_files = sorted(os.listdir(images_folder))
                annotation_files = sorted(os.listdir(annotations_folder))
                for img_name, ann_name in zip(image_files, annotation_files):
                    img_rel = f"images/{img_name}".replace("\\", "/")
                    ann_rel = f"annotations/{ann_name}".replace("\\", "/")
                    f.write(f"{img_rel} {ann_rel}\n")
            QMessageBox.information(self, "生成完成", f"lst.txt 已生成于 {lst_folder}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成lst.txt失败: {e}")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = DatasetFormatAdjuster()
    window.show()
    sys.exit(app.exec_())
