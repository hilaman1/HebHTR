from pdf2image import convert_from_path
import os


def pdf_to_image(pdf_path, jpg_path, pdf_name):
    """
    convert pdf to images and save them in the jpg_path folder
    :param pdf_path:  the path of the pdf file
    :param jpg_path:  the path of the jpg folder
    :param pdf_name:  the name of the pdf file
    """
    images = convert_from_path(pdf_path)
    for i in range(len(images)):
        images[i].save(fr'{jpg_path}\{pdf_name}_{i}.jpg', 'JPEG')


def main():
    # TODO - change the path to the project path
    project_path = r'C:\Users\hillu\OneDrive\מסמכים\nlp\final_project\HebHTR'
    data_path = fr'{project_path}\data'
    questionnaires_pdf_dir_path = fr'{data_path}\questionnaires pdf'
    questionnaires_jpg_dir_path = fr'{data_path}\questionnaires jpg'

    for pdf_file in os.listdir(questionnaires_pdf_dir_path):
        pdf_path = fr'{questionnaires_pdf_dir_path}\{pdf_file}'
        pdf_name = pdf_path.split('\\')[-1].split('.')[0]
        # check if there is a folder with the name of the pdf
        jpg_path = fr'{questionnaires_jpg_dir_path}\{pdf_name}'
        if not os.path.exists(fr'{jpg_path}'):
            os.mkdir(fr'{jpg_path}')
        # convert the pdf to images and save them in the folder
        pdf_to_image(pdf_path, jpg_path, pdf_name)


main()
