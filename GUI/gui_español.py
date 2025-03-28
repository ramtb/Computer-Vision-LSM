from PySide6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QWidget, QVBoxLayout, QLabel, 
                               QScrollArea, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QLineEdit,
                               QPushButton, QSpacerItem, QSizePolicy, QGroupBox, QGridLayout, QScrollArea, QFrame)
                               
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QColor, QPixmap, QPainter, QIcon, QMovie
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QSvgWidget
from modules.loaders import RelativeDirToRoot
import sys
### Path images ###
relative_dir = RelativeDirToRoot(root_dir='Computer-vision-LSM')
path_images = relative_dir.generate_path('GUI\\assets\\images')

class MainWindow(QMainWindow):
    def __init__(self,app):
        super().__init__()
        self.app = app
        self.setWindowTitle("Keet: Rompiendo barreras de silencio")
        self.resize(854, 480)
        appIcon = QIcon("GUI\\assets\\images\\keet.svg")
        self.setWindowIcon(appIcon)
        menu = self.menuBar()
        menu.addMenu("Reconocimiento de la LSM y su vocalización")

        self.btn1 = QPushButton("Iniciar", self)
        self.btn1.setToolTip("Runs the translation system")
        self.btn1.clicked.connect(self.open_GUI)
        self.btn1.setCursor(Qt.PointingHandCursor)

        self.btn2 = QPushButton("Glosario", self)
        self.btn2.setToolTip("See the sign language glossary")
        self.btn2.clicked.connect(self.open_glosario)
        self.btn2.setCursor(Qt.PointingHandCursor)
        # self.btn3 = QPushButton("Modelos de emociones", self)
        # self.btn3.clicked.connect(self.open_training)
        # self.btn3.setToolTip("Train facial expression recognition modelss")
        # self.btn3.setCursor(Qt.PointingHandCursor)
        self.btn4 = QPushButton("Creditos", self)
        self.btn4.setToolTip("View system credits")
        self.btn4.clicked.connect(self.open_creditos)
        self.btn4.setCursor(Qt.PointingHandCursor)
        self.btn5 = QPushButton("Salir", self)
        self.btn5.setToolTip("Exit the system")
        self.btn5.clicked.connect(self.quit_app)
        self.btn5.setCursor(Qt.PointingHandCursor)
        


        self.unamsvg = QSvgWidget("GUI\\assets\\images\\unam_naranja.svg", parent=self)
        self.ciencias = QSvgWidget("GUI\\assets\\images\\ciencias_naranja.svg", parent=self)
        self.keetsvg = QSvgWidget("GUI\\assets\\images\\keet.svg", parent=self)
        self.sekkansvg = QSvgWidget("GUI\\assets\\images\\sekkan.svg", parent=self)

        #self.svgWidget = QSvgWidget("images\\cat-face.svg", parent=self)


    def resizeEvent(self, event):
        """Este método se ejecuta cuando la ventana cambia de tamaño."""
        # Obtener el tamaño actual de la ventana
        ancho = self.width()
        alto = self.height()

        total_altura = 80*3 + 20*2
        inicio_y = (alto - total_altura) // 2
        #if ancho == 854:
            #alto_y = (alto*0.15)
        #else:
            #alto_y = (alto*0.1)
        #extra = 65
        # Calcular las posiciones dinámicas
        self.btn1.setGeometry((ancho - 200)//2, inicio_y, 200, 80)  # 10% de la ventana
        self.btn2.setGeometry((ancho - 200)//2, inicio_y + 100, 200, 80)  # 70% ancho, 10% alto
        # self.btn3.setGeometry((ancho - 200)//2, inicio_y + 200, 200, 80)  # 30% ancho, 50% alto
        self.btn4.setGeometry(int(ancho*0.01), int(alto * 0.99- 80), 200, 80)
        self.btn5.setGeometry(int(ancho*0.99 - 200), int(alto * 0.99- 80), 200, 80)
        
         # Centrado
        

        self.unamsvg.setGeometry(
            (ancho*0.01),  # Centrar horizontalmente
            40,   # Centrar verticalmente
            (ancho*0.1*0.89), (ancho*0.1)            # Mantener tamaño fijo
        )
        self.ciencias.setGeometry(
            (ancho*0.01 + ancho*0.1*0.89 + 10),  # Centrar horizontalmente
            40,   # Centrar verticalmente
            (ancho*0.1*0.89), (ancho*0.1)            # Mantener tamaño fijo
        )
        self.keetsvg.setGeometry(
            (ancho*0.99 - ancho*0.1),  # Centrar horizontalmente
            40,   # Centrar verticalmente
            (ancho*0.1), (ancho*0.1))            # Mantener tamaño fijo
        self.sekkansvg.setGeometry(
            (ancho*0.99 - ancho*0.1*0.8 -ancho*0.1 -10),  # Centrar horizontalmente
            40,   # Centrar verticalmente
            (ancho*0.1*0.8), (ancho*0.1))            # Mantener tamaño fijo
        # Continuar con el evento de redimensionar
        
        super().resizeEvent(event)

    def quit_app(self):
        self.app.quit()

    def show_fullscreen(self):
        """Mostrar la ventana en pantalla completa."""
        self.showFullScreen()
    def open_creditos(self):
        """Abrir el nuevo widget y ocultar la ventana principal."""
        self.new_widget = Creditos(self)  # Pasamos la referencia de la ventana principal
        self.new_widget.show()
        self.hide()
    def open_GUI(self):
        """Abrir el nuevo widget y ocultar la ventana principal."""
        self.gui = GUI(self)
        self.gui.show()
        self.hide()
    def open_glosario(self):
        """Abrir el nuevo widget y ocultar la ventana principal."""
        self.glosario = Glosario(self)
        self.glosario.show()
        self.hide()
    def open_training(self):
        """Abrir el nuevo widget y ocultar la ventana principal."""
        self.training = training(self)
        self.training.show()
        self.hide()
    
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

class Creditos(QWidget):
    def __init__(self,parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Creditos")
        self.resize(854, 480)
        appIcon = QIcon("GUI\\assets\\images\\keet.svg")
        self.setWindowIcon(appIcon)
        scrollarea = QScrollArea()
        scrollarea.setWidgetResizable(True)

        container = QWidget()
        scrollarea.setWidget(container)

        grid_layout = QGridLayout(container)

        people_data = [("GUI\\assets\\images\\cat.png", "Héctor Gerardo Martínez Fuentes, Faculty of Sciences UNAM, Biomedical Physics"),
                       ("GUI\\assets\\images\\cat.png", "Armando Huitzilt Rodríguez, Faculty of Sciences UNAM, Biomedical Physics"),
                       ("GUI\\assets\\images\\cat.png", "Diego Chairez Veloz, UAM Iztapalapa, Biomedical Engineering"),
                       ("GUI\\assets\\images\\cat.png", "Angel Ramses Tellez Becerra, Faculty of Sciences UNAM, Biomedical Physics"),
                       ("GUI\\assets\\images\\cat.png", "Arturo Arroyo Nuñez, Faculty of Sciences UNAM, Biomedical Physics")]
        for i, (image, name) in enumerate(people_data):
            photo_label = QLabel()
            pixmap = QPixmap(image)
            pixmap = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            photo_label.setPixmap(pixmap)
            photo_label.setAlignment(Qt.AlignCenter)

            name_label = QLabel(name)
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            grid_layout.addWidget(photo_label, i, 0)  # Columna de la izquierda (foto)
            grid_layout.addWidget(name_label, i, 1) 

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scrollarea)
        

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 10, 0, 10)

        back_button = QPushButton("Regresar") 
        back_button.clicked.connect(self.show_parent)
        back_button.setToolTip("Return to the main menu")
        back_button.setCursor(Qt.PointingHandCursor)
        close_app_button = QPushButton("Salir")
        close_app_button.clicked.connect(self.quit_app)
        close_app_button.setToolTip("Exit the system")
        close_app_button.setCursor(Qt.PointingHandCursor)
        back_button.setFixedSize(200,80)  # Ancho: 150 px, Alto: 50 px
        close_app_button.setFixedSize(200,80)  # Tamaño mínimo

        button_layout.addWidget(back_button)
        button_layout.addStretch()
        button_layout.addWidget(close_app_button)

        main_layout.addLayout(button_layout)


        

    def show_parent(self):
        """Mostrar la ventana principal y ocultar el widget actual."""
        self.parent.show()
        self.hide()

        
    def quit_app(self):
        QApplication.quit()

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

class GUI(QWidget):
    close_application = Signal()

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.initUI()
        appIcon = QIcon("GUI\\assets\\images\\keet.svg")
        self.setWindowIcon(appIcon)

    def initUI(self):
        # Configurar la ventana principal
        self.setWindowTitle('Keet: Rompiendo barreras de silencio')
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
        
        self.clear_history_button = QPushButton('Borrar historial', self)
        self.clear_history_button.clicked.connect(self.clear_history)
        self.clear_history_button.setToolTip("Delete the translation history")
        self.clear_history_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(self.clear_history_button)

        self.return_button = QPushButton('Regresar', self)
        self.return_button.clicked.connect(self.show_parent)
        self.return_button.setToolTip("Return to the main menu")
        self.return_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(self.return_button)

        self.close_app_button = QPushButton('Salir', self)
        self.close_app_button.clicked.connect(self.quit_app)
        self.close_app_button.setToolTip("Exit the system")
        self.close_app_button.setCursor(Qt.PointingHandCursor)
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
    
    def show_parent(self):
        """Mostrar la ventana principal y ocultar el widget actual."""
        self.parent.show()
        self.hide()

    def quit_app(self):
        QApplication.quit()

class Glosario(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Glossary of Mexican Sign Language")
        self.resize(854, 480)
        appIcon = QIcon("GUI\\assets\\images\\keet.svg")
        self.setWindowIcon(appIcon)
        self.main_layout = QVBoxLayout(self)
        self.ancho = self.width()
        self.alto = self.height()

        # Scroll Area para permitir desplazamiento
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        # Contenedor para el contenido scrollable
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)

        # Sección: Señas Estáticas
        static_label = QLabel("Static Signs")
        static_label.setFont(QFont("Helvetica Neue", 16, QFont.Bold))
        static_label.setAlignment(Qt.AlignCenter)
        static_label.setStyleSheet("color: #333; margin: 10px 0;")

        self.static_grid = QGridLayout()
        self.static_grid.setSpacing(10)

        # Añadir imágenes y etiquetas manualmente (Señas Estáticas)
        self.static_images = []
        self.add_static_image("A", "GUI\\assets\\images\\Static\\A.jpg", 0, 0)
        self.add_static_image("B", "GUI\\assets\\images\\Static\\B.jpg", 0, 1)
        self.add_static_image("C", "GUI\\assets\\images\\Static\\C.jpg", 0, 2)
        self.add_static_image("D", "GUI\\assets\\images\\Static\\D.jpg", 0, 3)
        self.add_static_image("E", "GUI\\assets\\images\\Static\\E.jpg", 0, 4)
        self.add_static_image("F", "GUI\\assets\\images\\Static\\F.jpg", 2, 0)
        self.add_static_image("G", "GUI\\assets\\images\\Static\\G.jpg", 2, 1)
        self.add_static_image("H", "GUI\\assets\\images\\Static\\H.jpg", 2, 2)
        self.add_static_image("I", "GUI\\assets\\images\\Static\\I.jpg", 2, 3)
        self.add_static_image("L", "GUI\\assets\\images\\Static\\L.jpg", 2, 4)
        self.add_static_image("M", "GUI\\assets\\images\\Static\\M.jpg", 4, 0)
        self.add_static_image("N", "GUI\\assets\\images\\Static\\N.jpg", 4, 1)
        self.add_static_image("O", "GUI\\assets\\images\\Static\\O.jpg", 4, 2)
        self.add_static_image("P", "GUI\\assets\\images\\Static\\P.jpg", 4, 3)
        self.add_static_image("R", "GUI\\assets\\images\\Static\\R.jpg", 4, 4)
        self.add_static_image("S", "GUI\\assets\\images\\Static\\S.jpg", 6, 0)
        self.add_static_image("T", "GUI\\assets\\images\\Static\\T.jpg", 6, 1)
        self.add_static_image("U", "GUI\\assets\\images\\Static\\U.jpg", 6, 2)
        self.add_static_image("V", "GUI\\assets\\images\\Static\\V.jpg", 6, 3)
        self.add_static_image("W", "GUI\\assets\\images\\Static\\W.jpg", 6, 4)
        self.add_static_image("Y", "GUI\\assets\\images\\Static\\Y.jpg", 8, 0)
        


        # Línea divisoria
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("border: 1px solid gray; margin: 20px 0;")
        self.scroll_layout.addWidget(static_label)
        self.scroll_layout.addLayout(self.static_grid)
        self.scroll_layout.addWidget(separator)

        # Sección: Señas Dinámicas
        dynamic_label = QLabel("Dynamic Signs")
        dynamic_label.setFont(QFont("Helvetica Neue", 16, QFont.Bold))
        dynamic_label.setAlignment(Qt.AlignCenter)
        dynamic_label.setStyleSheet("color: #333; margin: 10px 0;")

        self.dynamic_grid = QGridLayout()
        self.dynamic_grid.setSpacing(10)

        # Añadir GIFs dinámicos
        self.dynamic_gifs = []
        self.add_dynamic_gif("I", "GUI\\assets\\images\\cat.gif", 0, 0)
        self.add_dynamic_gif("J", "GUI\\assets\\images\\cat.gif", 0, 1)
        self.add_dynamic_gif("K", "GUI\\assets\\images\\cat.gif", 0, 2)
        self.add_dynamic_gif("L", "GUI\\assets\\images\\cat.gif", 0, 3)
        self.add_dynamic_gif("M", "GUI\\assets\\images\\cat.gif", 0, 4)
        self.add_dynamic_gif("N", "GUI\\assets\\images\\cat.gif", 2, 0)

        self.scroll_layout.addWidget(dynamic_label)
        self.scroll_layout.addLayout(self.dynamic_grid)

        # Configurar área de scroll
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)

        # Botones de navegación
        button_layout = QVBoxLayout()
        back_button = QPushButton("Regresar")
        back_button.setCursor(Qt.PointingHandCursor)
        back_button.setToolTip("Regresar al menú principal")
        exit_button = QPushButton("Salir")
        exit_button.setCursor(Qt.PointingHandCursor)
        exit_button.setToolTip("Exit the system")

        button_layout.addWidget(back_button)
        button_layout.addWidget(exit_button)
        self.main_layout.addLayout(button_layout)

        # Conectar botones
        back_button.clicked.connect(self.show_parent)  # Regresar cierra el widget actual
        exit_button.clicked.connect(QApplication.instance().quit)  # Salir cierra la aplicación completa

    def add_static_image(self, label_text, image_path, row, col):
        """Añade una imagen estática al grid."""
        image_label = QLabel()
        text_label = QLabel(label_text)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setFont(QFont("Helvetica Neue", 12, QFont.Bold))

        self.static_images.append((image_label, image_path))  # Guardar referencia
        self.static_grid.addWidget(image_label, row, col, Qt.AlignCenter)
        self.static_grid.addWidget(text_label, row + 1, col, Qt.AlignCenter)

    def add_dynamic_gif(self, label_text, gif_path, row, col):
        """Añade un GIF dinámico al grid."""
        gif_label = QLabel()
        text_label = QLabel(label_text)
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setFont(QFont("Helvetica Neue", 12, QFont.Bold))

        movie = QMovie(gif_path)
        gif_label.setMovie(movie)
        movie.start()

        self.dynamic_gifs.append((gif_label, movie))  # Guardar referencia del QLabel y el QMovie
        self.dynamic_grid.addWidget(gif_label, row, col, Qt.AlignCenter)
        self.dynamic_grid.addWidget(text_label, row + 1, col, Qt.AlignCenter)

    def resizeEvent(self, event):
        """Redimensiona las imágenes y GIFs al cambiar el tamaño de la ventana."""
        new_width = self.width() // 5
        new_height = self.height() // 5

        # Redimensionar imágenes estáticas
        for image_label, image_path in self.static_images:
            pixmap = QPixmap(image_path).scaled(new_width, new_height, Qt.KeepAspectRatio)
            image_label.setPixmap(pixmap)

        # Redimensionar GIFs dinámicos
        for gif_label, movie in self.dynamic_gifs:
            # Obtener dimensiones actuales del contenedor
            container_width = self.width() // 5
            container_height = self.height() // 5

            # Obtener dimensiones del GIF original
            original_size = movie.currentPixmap().size()
            original_width, original_height = original_size.width(), original_size.height()

            # Mantener proporción
            aspect_ratio = original_width / original_height
            if container_width / container_height > aspect_ratio:
                new_width = int(container_height * aspect_ratio)
                new_height = container_height
            else:
                new_width = container_width
                new_height = int(container_width / aspect_ratio)

            # Aplicar tamaño escalado
            movie.setScaledSize(QSize(new_width, new_height))

        super().resizeEvent(event)
    def show_parent(self):
        """Mostrar la ventana principal y ocultar el widget actual."""
        self.parent.show()
        self.hide()

class training(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Modelos de emociones")
        appIcon = QIcon("GUI\\assets\\images\\keet.svg")
        self.setWindowIcon(appIcon)
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        window_width = 420
        window_height = 600
        x_position = screen_width - window_width
        y_position = (screen_height - window_height) // 2 
        width = window_width//4
        self.setGeometry(x_position, y_position, window_width, window_height)
        emotion_emojis = {
            'ENOJO': '😠',
            'FELIZ': '😊',
            'NEUTRAL': '😐',
            'SORPRESA': '😲',
            'TRISTE': '😢'
        }
        self.unamsvg = QSvgWidget("GUI\\assets\\images\\unam_naranja.svg", parent=self)
        self.unamsvg.setGeometry(5, 10, width - 10, int(width * 1.13) - 10)

        self.ciencias = QSvgWidget("GUI\\assets\\images\\ciencias_naranja.svg", parent=self)
        self.ciencias.setGeometry(width, 10, width - 10, int(width * 1.13) - 10)

        self.sekkansvg = QSvgWidget("GUI\\assets\\images\\SEKKAN_slogan_naranja.svg", parent=self)
        self.sekkansvg.setGeometry(width * 2, 10, width * 2 - 10, width - 10)

        # Etiqueta principal
        self.instruction_label = QLabel("Seleccionar emociones ", self)
        self.instruction_label.setFont(QFont("Helvetica Neue", 14))
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setGeometry(10, 150, 400, 30)

        # Crear botones individuales para cada emoción
        self.enojo_button = QPushButton("😠", self)
        self.enojo_button.setFont(QFont("Helvetica Neue", 20))
        self.enojo_button.setGeometry(15, 200, 70, 70)
        self.enojo_button.setToolTip("Angry")
        self.enojo_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                font-size: 45px;
            }
            QPushButton:hover {
                background-color: #d6eaff;
            }
        """)

        self.feliz_button = QPushButton("😊", self)
        self.feliz_button.setFont(QFont("Helvetica Neue", 20))
        self.feliz_button.setGeometry(95, 200, 70, 70)
        self.feliz_button.setToolTip("Happy")
        self.feliz_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                font-size: 45px;
            }
            QPushButton:hover {
                background-color: #d6eaff;
            }
        """)
       

        self.neutral_button = QPushButton("😐", self)
        self.neutral_button.setFont(QFont("Helvetica Neue", 20))
        self.neutral_button.setGeometry(175, 200, 70, 70)
        self.neutral_button.setToolTip("Neutral")
        self.neutral_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                font-size: 45px;
            }
            QPushButton:hover {
                background-color: #d6eaff;
            }
        """)
        

        self.sorpresa_button = QPushButton("😲", self)
        self.sorpresa_button.setFont(QFont("Helvetica Neue", 20))
        self.sorpresa_button.setGeometry(255, 200, 70, 70)
        self.sorpresa_button.setToolTip("Surprised")
        self.sorpresa_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                font-size: 45px;
            }
            QPushButton:hover {
                background-color: #d6eaff;
            }
        """)
        
        

        self.triste_button = QPushButton("😢", self)
        self.triste_button.setFont(QFont("Helvetica Neue", 20))
        self.triste_button.setGeometry(335, 200, 70, 70)
        self.triste_button.setToolTip("Sad")
        self.triste_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                font-size: 45px;
            }
            QPushButton:hover {
                background-color: #d6eaff;
            }
        """)

        self.text_input = QLineEdit( self)
        self.text_input.setPlaceholderText("Escribe tu nombre aquí:")
        self.text_input.setGeometry(10, 300, 400, 40)
        self.text_input.setStyleSheet("""
            QLineEdit {
                border: none;
                border-bottom: 2px solid #cccccc; /* Color de la línea por defecto */
                background-color: #e0e0e0;       /* Color del fondo */
                padding: 5px;
            }
            QLineEdit:focus {
                border-bottom: 2px solid #00b56d; /* Cambia este color */
                background-color: #f0f0f0;
            }
        """)

        # Botón "Entrenar"
        self.train_button = QPushButton("Entrenar", self)
        self.train_button.setFont(QFont("Helvetica Neue", 14))
        self.train_button.setGeometry(150, 350, 120, 40)
        self.train_button.setCursor(Qt.PointingHandCursor)
        self.train_button.setToolTip("Entrenar el modelo")

        self.close_button = QPushButton("Regresar", self)
        self.close_button.setGeometry(10, 540, 100, 50)
        self.close_button.clicked.connect(self.show_parent)
        self.close_button.setToolTip("Return to the main menu")
        self.close_button.setCursor(Qt.PointingHandCursor)


        self.exit_button = QPushButton("Salir", self)
        self.exit_button.setGeometry(310, 540, 100, 50)
        self.exit_button.clicked.connect(QApplication.instance().quit)
        self.exit_button.setToolTip("Exit the system")
        self.exit_button.setCursor(Qt.PointingHandCursor)

        

    def show_parent(self):
        """Mostrar la ventana principal y ocultar el widget actual."""
        self.parent.show()
        self.hide()

    def quit_app(self):
        QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    with open("GUI\\assets\\style.qss", "r") as qss_file:
        qss_style = qss_file.read()
    app.setStyleSheet(qss_style)
    gui = MainWindow(app)
    gui.show()
    app.exec()


