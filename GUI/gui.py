from PySide6.QtWidgets import (QApplication, QHBoxLayout, QWidget, QVBoxLayout, QLabel, 
                               QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
                               QPushButton, QSpacerItem, QSizePolicy, QGroupBox
                               )
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer

### Path images ###
path_images = "C://Users//arhui//Documents//projects//keet//src//images"


class GUI(QWidget):
    close_application = Signal()

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Configurar la ventana principal
        self.setWindowTitle('Keet: Rompiedo barreras del silencio')
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        window_width = 420
        window_height = 600
        x_position = screen_width - window_width
        y_position = (screen_height - window_height) // 2 

        self.setGeometry(x_position, y_position, window_width, window_height)
        
        # Layout principal
        main_layout = QVBoxLayout()

        # Layout con logos
        logo_layout = QHBoxLayout()

        left_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        logo1_label = QLabel(self)
        logo2_label = QLabel(self)
        logo3_label = QLabel(self)

        self.load_svg_to_label(f"{path_images}//SEKKAN_slogan_naranja.svg", logo1_label, 205, 95)  # Tamaño reducido
        self.load_svg_to_label(f"{path_images}//UNAM_negro.svg", logo2_label, 70, 87)
        self.load_svg_to_label(f"{path_images}//Ciencias_negro.svg", logo3_label, 73, 95)

        right_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        logo_layout.setContentsMargins(0, 0, 0, 0)  # Sin márgenes
        logo_layout.setSpacing(20)  # Espacio reducido

        logo_layout.addItem(left_spacer)
        logo_layout.addWidget(logo1_label)
        logo_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        logo_layout.addWidget(logo2_label)
        logo_layout.addWidget(logo3_label)
        logo_layout.addItem(right_spacer)

        main_layout.addLayout(logo_layout)

        main_layout.addItem(QSpacerItem(20, 20))

        # Crear área de historial
        history_group_box = QGroupBox("Historial de traducciones", self)

        history_layout = QVBoxLayout()

        self.history_area = QLabel('', self)
        self.history_area.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.history_area.setStyleSheet("font-size: 15px; ")
        self.history_area.setWordWrap(True)
        self.history_area.setFixedHeight(100)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.history_area)
        scroll_area.setMaximumHeight(100) 

        history_layout.addWidget(scroll_area)

        history_group_box.setLayout(history_layout)

        main_layout.addWidget(history_group_box)

        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.main_label = QLabel('Esperando ...', self)
        self.main_label.setAlignment(Qt.AlignCenter)
        self.main_label.setStyleSheet("font-size: 35px; ")

        font = QFont("Segoe UI Emoji", 24)
        self.main_label.setFont(font)
        
        # Indicador LED
        self.led_view = QGraphicsView(self)
        self.led_scene = QGraphicsScene(self)
        self.led_view.setScene(self.led_scene)
        self.led_view.setFixedSize(30, 30)
        self.led_item = QGraphicsEllipseItem(0, 0, 25, 25)
        self.led_item.setBrush(QColor(200, 0, 0))  
        self.led_scene.addItem(self.led_item)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.led_view)
        h_layout.addWidget(self.main_label)

        main_layout.addLayout(h_layout)

        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        button_layout = QHBoxLayout()
        
        self.clear_history_button = QPushButton('Eliminar historial', self)
        self.clear_history_button.clicked.connect(self.clear_history)
        button_layout.addWidget(self.clear_history_button)

        self.close_app_button = QPushButton('Cerrar aplicación', self)
        self.close_app_button.clicked.connect(self.close_application.emit)
        button_layout.addWidget(self.close_app_button)


        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)


    def load_svg_to_label(self, svg_path, label, width, height):
        """
        Cargar un archivo SVG y renderizarlo dentro de un QLabel.
        """
        renderer = QSvgRenderer(svg_path)
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)  
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        label.setPixmap(pixmap)

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

if __name__ == "__main__":
    app = QApplication([])
    with open("style.qss", "r") as qss_file:
        qss_style = qss_file.read()
    app.setStyleSheet(qss_style)
    gui = GUI()
    gui.show()
    app.exec()
