import os
from DefaultConfig import DefaultConfig
from modules.LoadImage import LoadImage
from modules.EggModelGen import EggModelGen

class Matrix2x2Model(DefaultConfig):

    def __init__(self):
        super().__init__()

    
    def load_train_images(self):
        paths = []
        folder_path = os.path.join(os.getcwd(), "images", "matrix_unique_type", "2x2")

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, file_name)
                paths.append(image_path)

        return paths
    
    def load_test_images(self):
        paths = []
        folder_path = os.path.join(os.getcwd(), "images", "treated_images")

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, file_name)
                paths.append(image_path)

        return sorted(paths, key=lambda i: int(i.split("sample_")[1].split(".")[0]))
    
    def create_instruction(self):
        with open(os.path.join(os.getcwd(), "models", "prompts", "matrix_2x2.txt"), 'r', encoding='utf-8') as file:
            linhas = "".join(file.readlines())
        return linhas



def main():
    """Função principal para executar o fluxo de trabalho."""


    model = Matrix2x2Model()


    images_path = model.load_train_images()

    train_files_gemini = []
    for image in images_path:
        file_gemini = model.load_image_to_gemini(image)

        train_files_gemini.append(file_gemini)
    
    test_paths = model.load_test_images()

    prompt = model.create_instruction()

    #Aguardar o upload
    LoadImage.wait_for_files_active(train_files_gemini)

    temperatures = [
        # 0, 
        # 0.5, 
        1 , 
        # 1.5, 
        # 2
        ]
    
    # if (not os.path.exists(os.path.join(os.getcwd(), "results", "2x2"))):

    #     os.makedirs(os.path.join(os.getcwd(), "results", "2x2"))

    #     model.result_dataframe.to_csv(os.path.join(os.getcwd(), "results", "2x2", "result.csv"), sep=";", index=False)

    for temperature in temperatures:

        if (not os.path.isdir(os.path.join(os.getcwd(), "results", "2x2", f"temperature {temperature}"))):
            os.makedirs(os.path.join(os.getcwd(), "results", "2x2", f"temperature {temperature}"))

        egg_ia_gen = EggModelGen(temperature=temperature)

        egg_ia_gen.create_model(initial_instruction=prompt)

        examples = train_files_gemini[:]
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


            result_df.iloc[:20,:].to_csv(os.path.join(os.getcwd(), "results", "2x2", f"temperature {temperature}", f"result_{(i*20) + 1}_to_{((i+1)*20)}.csv"), sep=";",index=False)


if __name__ == "__main__":
    main()