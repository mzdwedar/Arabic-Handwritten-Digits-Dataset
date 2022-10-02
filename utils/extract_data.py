from zipfile import ZipFile

if __name__ == '__main__':
    files = ['Train + Test Matlab.mat', 'csvTestLabel 10k x 1.csv', 'csvTrainLabel 60k x 1.csv']

    with ZipFile('ahdd1.zip', 'r') as zipObj:
        for file in files:
            zipObj.extract(file)