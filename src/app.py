from views.main_ui import QtWidgets, Ui_MainWindow
from perceptron import Perceptron

class App(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        
        self.fileDialog = QtWidgets.QFileDialog(self)
        self.fileDialog.setNameFilter("Archivos CSV (*.csv, *.CSV)")
        self.file_btn.clicked.connect(self.select_file)
        self.fileDialog.fileSelected.connect(self.update_filename)
        self.fileDialog.rejected.connect(self.reset_filename)
        self.default_filename_text = "Seleccione archivo"
        self.filename.setText(self.default_filename_text)
        self.file_path = None
        self.output.setText("")
        self.start.clicked.connect(self.run)
        
    def select_file(self):
        self.fileDialog.show()
        
    def update_filename(self, filename):
        self.filename.setText(filename.split("/")[-1])
        self.file_path = filename
        
    def reset_filename(self):
        self.filename.setText(self.default_filename_text)
        self.file_path = None
        
    def run(self):
        unique_instance = Perceptron(
            self.file_path,
            int(self.iterations.text()),
            float(self.eta.text())
        )
        output_message = unique_instance.run()
        self.output.setText(output_message)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = App()
    window.show()
    app.exec_()