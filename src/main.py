from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Configurar la ventana principal
        self.setWindowTitle('Reconocimiento de Gestos')
        self.setGeometry(100, 100, 800, 600)
        
        # Crear layout
        layout = QVBoxLayout()

        # Crear display de texto
        self.label = QLabel('Esperando predicciones...', self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 35px;")

        # Configurar una fuente que soporte emojis
        font = QFont("Segoe UI Emoji", 24)
        self.label.setFont(font)
        
        # Añadir el display al layout
        layout.addWidget(self.label)

        # Establecer layout
        self.setLayout(layout)

    def update_text(self, text):
        """
        Actualiza el display de texto con la predicción recibida.
        """
        self.label.setText(text)

