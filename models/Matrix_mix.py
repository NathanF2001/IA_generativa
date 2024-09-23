import os
from DefaultConfig import DefaultConfig
from modules.LoadImage import LoadImage
from modules.EggModelGen import EggModelGen
import pandas as pd

class MatrixMixModel(DefaultConfig):

    def __init__(self):
        super().__init__()

    
    def load_train_images(self):
        paths = []
        folder_path = os.path.join(os.getcwd(), "images", "matrix_mixed_type", "5x4")

        for i in range(1,6):
            image_path = os.path.join(folder_path, f"mixed_type_{i}.jpg")
            result = self.read_csv(os.path.join(folder_path, f"reference_{i}.csv"))
            paths.append([image_path, result])    

        return paths
    
    def load_test_images(self):
        paths = []
        folder_path = os.path.join(os.getcwd(), "images", "treated_images")

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, file_name)
                paths.append(image_path)

        return sorted(paths, key=lambda i: int(i.split("sample_")[1].split(".")[0]))
    
    def create_instruction(self,prompt_file):
        with open(os.path.join(os.getcwd(), "models", "prompts", prompt_file), 'r', encoding='utf-8') as file:
            linhas = "".join(file.readlines())
        return linhas
    
    def read_csv(self, file):
        with open(file, 'r', encoding='utf-8') as file:
            linhas = "".join(file.readlines())
        return linhas



def main(prompt_file='matrix_mix.txt',isLowPrompt=False):
    """Função principal para executar o fluxo de trabalho."""


    model = MatrixMixModel()


    images_path = model.load_train_images()

    train_prompt = []
    train_files_gemini = []
    for image, result in images_path:
        file_gemini = model.load_image_to_gemini(image)

        train_prompt.append(file_gemini)
        train_files_gemini.append(file_gemini)
        train_prompt.append(result)
    
    test_paths = model.load_test_images()

    prompt = model.create_instruction(prompt_file)

    #Aguardar o upload
    LoadImage.wait_for_files_active(train_files_gemini)

    temperatures = [
        0, 
        0.5, 
        1 , 
        1.5, 
        2
        ]
    
    # if (not os.path.exists(os.path.join(os.getcwd(), "results", "2x2"))):

    #     os.makedirs(os.path.join(os.getcwd(), "results", "2x2"))

    #     model.result_dataframe.to_csv(os.path.join(os.getcwd(), "results", "2x2", "result.csv"), sep=";", index=False)

    for temperature in temperatures:

        if (not os.path.isdir(os.path.join(os.getcwd(), "results", "5x4", f"temperature {temperature}"))):
            os.makedirs(os.path.join(os.getcwd(), "results", "5x4", f"temperature {temperature}"))

        if (not os.path.isdir(os.path.join(os.getcwd(), "results", "5x4", f"temperature {temperature}", 'low'))):
            os.makedirs(os.path.join(os.getcwd(), "results", "5x4", f"temperature {temperature}",'low'))

        if (not os.path.isdir(os.path.join(os.getcwd(), "results", "5x4", f"temperature {temperature}", 'high'))):
            os.makedirs(os.path.join(os.getcwd(), "results", "5x4", f"temperature {temperature}",'high'))

        egg_ia_gen = EggModelGen(temperature=temperature)

        egg_ia_gen.create_model(initial_instruction=prompt)

        examples = train_prompt[:]
        chat_session = egg_ia_gen.model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": examples,
                },
            ]
        )

        for i,test_path in enumerate(test_paths):

            print( f"Temperature {temperature}: File ", i+1)

            gemini_path = model.load_image_to_gemini(test_path)

            LoadImage.wait_for_files_active([gemini_path])

            response = chat_session.send_message(gemini_path)

            result_df = model.result_to_dataframe(response.text)


            result_df.iloc[:20,:].to_csv(os.path.join(os.getcwd(), "results", "5x4", f"temperature {temperature}", 'low' if isLowPrompt else 'high', f"result_{(i*20) + 1}_to_{((i+1)*20)}.csv"), sep=";",index=False)

        results = sorted(os.listdir(f"./results/5x4/temperature {temperature}/{'low' if isLowPrompt else 'high'}"), key=lambda x: int(x.split("_")[1]))
        df = pd.DataFrame()

        for i, path in enumerate(results):

            df_result_range = pd.read_csv(f"results/5x4/temperature {temperature}/{'low' if isLowPrompt else 'high'}/{path}" ,sep =";")


            df = pd.concat([df, df_result_range])

        df.to_csv(f'./results/{'low' if isLowPrompt else 'high'}_prompt_mixed_temperature{temperature}.csv', sep=";", index=False)


import sys
if __name__ == "__main__":
    [_, file_prompt, size_prompt] = sys.argv
    print(size_prompt=='high')
    main(file_prompt, size_prompt == 'low')

    