import pandas as pd


def preprocessing_text(file_path):
    
    """
    .xlsx .xls ...
    """
    
    data = pd.read_excel(file_path)
    df = data.copy()
    df[['사람문장1', '사람문장2', '사람문장3']] = df[['사람문장1', '사람문장2', '사람문장3']].fillna('')
    df['사람문장'] = df['사람문장1'].astype(str) + df['사람문장2'].astype(str) + df['사람문장3'].astype(str)
    df = df[['사람문장', '감정_대분류']].rename(columns = {'사람문장' : 'Text', '감정_대분류' : 'Label'})
    
    df.loc[(df['Label'] == '불안'), 'Label'] = 0
    df.loc[(df['Label'] == '분노'), 'Label'] = 1
    df.loc[(df['Label'] == '상처'), 'Label'] = 2
    df.loc[(df['Label'] == '슬픔'), 'Label'] = 3
    df.loc[(df['Label'] == '당황'), 'Label'] = 4
    df.loc[(df['Label'] == '기쁨'), 'Label'] = 5
    
    train_data_list = []

    for data, label in zip(df['Text'], df['Label']):
        train_data = []
        train_data.append(data)
        train_data.append(str(label))
        
        train_data_list.append(train_data)
        
        
    return train_data_list

