from PySide6.QtWidgets import (QApplication, QHBoxLayout, QWidget, QVBoxLayout, QLabel, 
                               QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
                               QPushButton
                               )
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor


class GUI(QWidget):
    close_application = Signal()

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Configurar la ventana principal
        self.setWindowTitle('Reconocimiento de Gestos')
        self.setGeometry(100, 100, 800, 600)
        
        # Crear layout
        main_layout = QVBoxLayout()

        # Crear área de historial
        self.history_area = QLabel('', self)
        self.history_area.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.history_area.setStyleSheet("font-size: 15px;")
        self.history_area.setWordWrap(True)

        # Crear un QScrollArea para el historial
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.history_area)
        scroll_area.setMaximumHeight(100)  # Ajusta esto según tus necesidades
        main_layout.addWidget(scroll_area)

        # Crear display de texto principal
        self.main_label = QLabel('Esperando predicciones...', self)
        self.main_label.setAlignment(Qt.AlignCenter)
        self.main_label.setStyleSheet("font-size: 35px;")

        # Configurar una fuente que soporte emojis
        font = QFont("Segoe UI Emoji", 24)
        self.main_label.setFont(font)
        
        # Crear indicador LED
        self.led_view = QGraphicsView(self)
        self.led_scene = QGraphicsScene(self)
        self.led_view.setScene(self.led_scene)
        self.led_view.setFixedSize(30, 30)
        self.led_item = QGraphicsEllipseItem(0, 0, 20, 20)
        self.led_item.setBrush(QColor(200, 0, 0))  # Inicialmente rojo (apagado)
        self.led_scene.addItem(self.led_item)

        # Crear layout horizontal para el LED y el texto principal
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.led_view)
        h_layout.addWidget(self.main_label)

        # Añadir el layout horizontal al layout principal
        main_layout.addLayout(h_layout)

        # Crear botones
        button_layout = QHBoxLayout()
        
        self.clear_history_button = QPushButton('Eliminar historial', self)
        self.clear_history_button.clicked.connect(self.clear_history)
        button_layout.addWidget(self.clear_history_button)

        self.close_app_button = QPushButton('Cerrar aplicación', self)
        self.close_app_button.clicked.connect(self.close_application.emit)
        button_layout.addWidget(self.close_app_button)

        # Añadir layout de botones al layout principal
        main_layout.addLayout(button_layout)

        # Establecer layout
        self.setLayout(main_layout)
    def update_text(self, current_text, completed_text=None):
        """
        Actualiza el display de texto principal y el historial si se proporciona.
        """
        self.main_label.setText(current_text)
        
        if completed_text:
            current_history = self.history_area.text()
            new_history = f"{completed_text}\n{current_history}"
            self.history_area.setText(new_history)

    def update_led_status(self, is_active):
        """
        Actualiza el estado del LED.
        """
        if is_active:
            self.led_item.setBrush(QColor(0, 255, 0))  # Verde (encendido)
        else:
            self.led_item.setBrush(QColor(200, 0, 0))  # Rojo (apagado)

    def clear_history(self):
        """
        Limpia el historial de traducciones.
        """
        self.history_area.setText('')