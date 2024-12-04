import os
from tensorflow.keras.models import load_model
from joblib import load
import h5py as h5


class RelativeDirToRoot:
    def __init__(self, root_dir: str = 'Computer-vision-LSM'):
        self.root_dir = root_dir

    def _get_relative_position_to_root(self):
        working_directory = os.getcwd()
        working_dir_separated = working_directory.split('\\')
        try:
            workin_dir_position = working_dir_separated.index(self.root_dir)
            relative_position_to_root = abs(workin_dir_position - len(working_dir_separated)) - 1
            return relative_position_to_root
        except ValueError:
            raise ValueError(f"Root directory '{self.root_dir}' not found in the path: {working_directory}")

    def _generate_model_paths(self, relative_position_to_root, model_name, scaler_name, model_type):
        """
        Private method to generate paths for the model and scaler based on the relative position to the root.
        This method is encapsulated and should only be used inside the child classes.
        """
        model_dir = f'models\\Trained\\{model_type}'
        model_dir = ''.join(['..\\' for _ in range(relative_position_to_root)]) + model_dir
        model_path = os.path.join(model_dir, model_name)
        scaler_path = os.path.join(model_dir, scaler_name)
        return model_path, scaler_path
    
    def _generate_h5_paths(self, relative_position_to_root, h5_file):
        """
        Private method to generate paths for the h5 file based on the relative position to the root.
        This method is encapsulated and should only be used inside the child classes.
        """
        h5_dir = f'data\\features\\{h5_file}'
        h5_dir = ''.join(['..\\' for _ in range(relative_position_to_root)]) + h5_dir
        h5_path = h5_dir
        return h5_path





class ModelLoaderSigns(RelativeDirToRoot):
    """
    Class to manage the loading of models and scalers for signs.
    """
    def __init__(self,model_name: str, scaler_name: str, root_dir: str = 'Computer-vision-LSM' ):
        self.model_name = model_name
        self.scaler_name = scaler_name

        super().__init__(root_dir)
        relative_position_to_root = self._get_relative_position_to_root()
        
        self.model_path_signs, self.scaler_path_signs = self._generate_model_paths(
            relative_position_to_root, model_name, scaler_name, 'signs')

    def load_sign_model(self):
        if not os.path.exists(self.model_path_signs):
            raise FileNotFoundError(f"Model not found at path: {self.model_path_signs}")
        self.model = load_model(self.model_path_signs)
        return self.model

    def load_sign_scaler(self):
        if not os.path.exists(self.scaler_path_signs):
            raise FileNotFoundError(f"Scaler not found at path: {self.scaler_path_signs}")
        self.scaler = load(self.scaler_path_signs)
        return self.scaler





class ModelLoaderFace(RelativeDirToRoot):
    """
    Class to manage the loading of models and scalers for face.
    """
    def __init__(self, model_name: str, scaler_name: str, root_dir: str = 'Computer-vision-LSM'):
        self.model_name = model_name
        self.scaler_name = scaler_name
        
        super().__init__(root_dir)
        relative_position_to_root = self._get_relative_position_to_root()
        
        self.model_path_faces, self.scaler_path_faces = self._generate_model_paths(
            relative_position_to_root, model_name, scaler_name, 'face')

    def load_face_model(self):
        if not os.path.exists(self.model_path_faces):
            raise FileNotFoundError(f"Model not found at path: {self.model_path_faces}")
        self.model = load_model(self.model_path_faces)
        return self.model

    def load_face_scaler(self):
        if not os.path.exists(self.scaler_path_faces):
            raise FileNotFoundError(f"Scaler not found at path: {self.scaler_path_faces}")
        self.scaler = load(self.scaler_path_faces)
        return self.scaler
    
    
    
    
    
    
    
    
    
class load_dataset(RelativeDirToRoot):
        """ 
        Class to load h5 files
        """
        def __init__(self, h5_file: str ,root_dir = 'Computer-vision-LSM'):
            super().__init__(root_dir)
            self.h5_file = h5_file
            relative_position_to_root = self._get_relative_position_to_root()
            self.h5_path = self._generate_h5_paths(relative_position_to_root, h5_file)
            
            
        def load_h5(self):
            if not os.path.exists(self.h5_path):
                raise FileNotFoundError(f"data not found at path: {self.h5_path}")
            self.h5_file = h5.File(self.h5_path, 'r')
            return self.h5_file
            
        
            
