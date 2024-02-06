import numpy as np
import pandas as pd
import matplotlib.pyplot as pl


class Perceptron():
    def __init__(self, path_file, iterations_number, learnig_rate) -> None:
        
        self.path_file = path_file
        self.iterations_number = iterations_number
        self.learnig_rate = learnig_rate
        
        self.iterations_errors = []
        self.weights_records = []
        
        self.final_weights = None
        self.initials_weights = None
    
        self.permisible_error = 0

    def run(self):
        if self.path_file is None:
            return "Seleccione un archivo."
        if self.iterations_number == 0:
            return "Incremente el número de iteraciones."
        
        data = self.read_csv(self.path_file)
        columns_count = len(data.columns)
        
        current_weight = np.random.uniform(low=0,high=1,size=(columns_count,1)).round(4)
        input_columns = np.hstack([data.iloc[:, :-1].values, np.ones((data.shape[0], 1))])
        output_column = np.array(data.iloc[:, -1])
        
        self.initials_weights = current_weight
        self.fill_weight_record(columns_count)
        
        for _ in range(self.iterations_number):
            u = np.dot(input_columns, current_weight)
            predicted_output = np.where(u >= 0, 1, 0).reshape(-1, 1)
            errors = output_column.reshape(-1, 1) - predicted_output

            norm_error = np.linalg.norm(errors)
            self.iterations_errors.append(norm_error)

            for i in range(columns_count):
                self.weights_records[i].append(current_weight[i, 0])

            errors_product = np.dot(input_columns.T, errors)
            delta_w = self.learnig_rate * errors_product
            current_weight += delta_w

        self.final_weights = current_weight
        
        self.show_graphics()
        self.show_resume()
        return "¡Hecho!"
    
    def read_csv(self, path):
        return pd.read_csv(path, delimiter=";", header=None)
    
    def fill_weight_record(self, columns_number):
        for _ in range(columns_number):
            self.weights_records.append([])

    def show_graphics(self):
        pl.figure(figsize=(8, 10))
        
        pl.subplot(2, 1, 1)
        pl.plot(range(1, len(self.iterations_errors) + 1), self.iterations_errors)
        pl.title('Norma del Error')
        pl.xlabel('Iteración')
        pl.ylabel('Norma de la Iteración')

        pl.subplot(2, 1, 2)
        for i, epoch_weights in enumerate(self.weights_records[:-1]):
            pl.plot(range(1, len(epoch_weights) + 1), epoch_weights, label=f'Peso {i + 1}')
        pl.title('Evolución de pesos')
        pl.xlabel('Iteración')
        pl.ylabel('Peso')
        pl.legend()

        pl.tight_layout()
        pl.show()

    def show_resume(self):
        print(f'\n-#-#- RESUMEN DE LA EJECUCIÓN -#-#-\n\t* Pesos Iniciales:\n {self.initials_weights}\n\t* Tasa de aprendizaje: {self.learnig_rate}\n\t* Error Permisible: {self.permisible_error}\n\t* Iteraciones totales: {self.iterations_number}')