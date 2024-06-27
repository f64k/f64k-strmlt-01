from typing import Union, NamedTuple
import io,os,re,sys,math,time,uuid,ctypes,pickle,random,shutil,string,urllib,decimal,datetime,itertools,traceback,collections,statistics
import numpy as np, pandas as pd
import plotly.express as px
import huggingface_hub

import sklearn #, statsmodels
from sklearn import svm, neighbors, naive_bayes, neural_network, tree, ensemble, linear_model, discriminant_analysis, gaussian_process, manifold, cluster
#from statsmodels.tsa import seasonal

os.makedirs(".temp", exist_ok=True) # for temporary local files

class HfRepo(NamedTuple):
    repo_id: str
    repo_type: str
    token: str


### remove decoration and popup menu button at top
STYLE_CORRECTION = " ".join([
    "<style>",
    "header[data-testid='stHeader'] { display:none }",
    "div[data-testid='stSidebarHeader'] { display:none }",
    "div[data-testid='stAppViewBlockContainer'] { padding:1em }",
    "div[data-testid='collapsedControl'] { background-color:#EEE }",
    "a[href='https://streamlit.io/cloud'] { display:none }"
    "</style>"
])

###
def pandas_info(df: pd.DataFrame) -> Union[pd.DataFrame,str]:
    buffer = io.StringIO()
    df.info(buf=buffer)
    str_info = buffer.getvalue()
    try:
        lines = str_info.splitlines()
        df = (pd.DataFrame([x.split() for x in lines[5:-2]], columns=lines[3].split()).drop('Count',axis=1).rename(columns={'Non-Null':'Non-Null Count'}))
        return df
    except Exception as ex:
        print(ex)
        return str_info

### случайные числа, для отладки например
def df_random_dataframe(n_cols:int = 15, n_rows:int = 100) -> pd.DataFrame:
    df = pd.DataFrame(np.random.randn(n_rows, n_cols), columns=(f"col {i}" for i in range(n_cols)))
    return df

### обработка столбца V для дальнейшего удобства + столб T типа время
def df_process_v_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index() #
    df.rename(columns = {"index": "T"}, inplace=True)
    df["Vis"] = df.V.map(lambda v: 0 if str(v)=="nan" else 1).astype(int)
    df["Vfloat"] = df.V.map(lambda v: 0 if str(v)=="nan" else str(v).replace(',', '.')).astype(float)
    df["Vsign"] = df.Vfloat.map(lambda v: -1 if v<0 else 1 if v>0 else 0).astype(int)
    df["Vposneg"] = df.Vfloat.map(lambda v: "n" if v<0 else "p" if v>0 else "o").astype(str)
    return df

###
def save_dataframe_to_hf(repo: HfRepo, dfToSave: pd.DataFrame, new_filename: str, remote_subdir: str) -> Union[huggingface_hub.CommitInfo, Exception]:
    """ save dataframe to hf repo """
    try:
        local_filename = os.path.join(".temp", new_filename)
        #df.to_csv('compressed_data.zip', index=False, compression={'method': 'zip', 'archive_name': 'data.csv'})
        dfToSave.to_csv(local_filename, index=False, sep=";", encoding="utf-8") # , compression="zip"
        apiHF = huggingface_hub.HfApi(token=repo.token)
        path_in_repo = os.path.basename(local_filename)
        if remote_subdir:
            path_in_repo = f"{remote_subdir}/{path_in_repo}"
        commit_info = apiHF.upload_file(path_or_fileobj=local_filename, path_in_repo=path_in_repo, repo_id=repo.repo_id, repo_type=repo.repo_type)
        return commit_info
    except Exception as exSave:
        return exSave


###
def load_dataframes_from_hf(repo: HfRepo, lstCsvFiles: list[str] = []) -> {str, pd.DataFrame}:
    """ load dataframes from hf """
    #https://huggingface.co/datasets/f64k/gaziev/blob/main/TestData3_2204_noAB_gaziev.zip
    dict_res = {}
    for fl_name in lstCsvFiles:
        try: file_loaded = huggingface_hub.hf_hub_download(filename=fl_name, repo_id=repo.repo_id, repo_type=repo.repo_type, token=repo.token)
        except: file_loaded = ""
        if os.path.exists(file_loaded):
            compress = "zip" if file_loaded.lower().endswith("zip") else None
            df_loaded = pd.read_csv(file_loaded, sep=";", encoding = "utf-8", compression=compress)
            dict_res[fl_name] = df_loaded # df_Vproc = df_process_v_column(df_loaded)
    return dict_res

### список CSV и ZIP файлов (c уровнем вложенности) в репозитории
### https://huggingface.co/docs/huggingface_hub/en/guides/hf_file_system
def list_files_hf(repo: HfRepo) -> list[str]:
    """ List CSV and ZIP files in HF repo """
    fs = huggingface_hub.HfFileSystem(token=repo.token)
    path_hf = f"{repo.repo_type}s/{repo.repo_id}/"
    #lst = fs.ls(path_hf, detail=False)
    lstGlob = fs.glob(path_hf + "**") # map(os.path.basename, lstGlob)
    lstNames = [fname.replace(path_hf, "") for fname in lstGlob if fname.lower().endswith(".csv") or fname.lower().endswith(".zip")]
    return lstNames

###
def plotly_xyzv_scatter_gray(df3D):
    """ 3D plot """
    color_discrete_map = dict(o='rgb(230,230,230)', p='rgb(90,1,1)', n='rgb(1,1,90)')
    fig = px.scatter_3d(df3D, x='X', y='Y', z='Z', color="Vposneg", opacity=0.4, height=800, color_discrete_map=color_discrete_map)
    fig.update_scenes(
        xaxis={"gridcolor":"rgba(30, 0, 0, 0.2)","color":"rgb(100, 0, 0)","showbackground":False},
        yaxis={"gridcolor":"rgba(0, 30, 0, 0.2)","color":"rgb(0, 100, 0)","showbackground":False},
        zaxis={"gridcolor":"rgba(0, 0, 30, 0.2)","color":"rgb(0, 0, 100)","showbackground":False})
    fig.update_traces(marker_size=3)
    return fig


# lstRepoZipFiles = ["TrainData_1504_AB_gaziev.zip","TestData_1504_AB_gaziev.zip","TestData3_2204_noAB_gaziev.zip"]
### returns (classifier_object, df_train_with_predict, time_elapsed)
def GetClassifier(lstDfOriginal, nHystorySteps) :
    #lstDfOriginal = [df_9125_Train, df_12010_Test, df_9051_Test3]
    nShift = nHystorySteps
    nCurrShift = nHystorySteps
    classifierName = "DecisionTreeClassifier"
    colsVectorInp = ["X","Y","Z"]
    fieldY = "Vis" #
    lstDataFrames = MakeHystoryColumns(lstDfOriginal, nShift)
    df_train = pd.concat(lstDataFrames)
    lstColsShift = [f"{c}-{i}" for i in range(1, nCurrShift+1) for c in colsVectorInp] # для nCurrShift=0 lstColsShift=[]
    colsVectorInpAll = colsVectorInp + lstColsShift
    y_train = df_train[fieldY]
    x_train_vect = df_train[colsVectorInpAll]
    dictClassifiers = createDictClassifiers_BestForXYZ()
    classifierObject = dictClassifiers[classifierName]
    start2 = time.time()
    classifierObject.fit(x_train_vect, y_train) # процесс обучения
    time_elapsed = time.time() - start2
    y_pred = classifierObject.predict(x_train_vect)
    df_train[f"predict_{fieldY}"] = y_pred
    print(f"{time_elapsed=}")
    return (classifierObject, df_train, time_elapsed)

#
def MakeHystoryColumns(lstDfOriginal, nShift) :
    lstDataframesShifted = [df.copy() for df in lstDfOriginal]
    lstColsShift = []
    for i in range(1, nShift+1):
        #cols = ["X","Y","Z"]+["A","B"]
        cols = ["X","Y","Z"]
        #cols = ["A","B"]
        for c in cols:
            for dfShift in lstDataframesShifted:
                dfShift[f'{c}-{i}'] = dfShift[c].shift(i).fillna(0)
            lstColsShift.append(lstDataframesShifted[0].columns[-1])
    print(lstColsShift)
    return lstDataframesShifted

RANDOM_STATE=11

def createDictClassifiers_BestForXYZ() :
    dictFastTree = {
        #"RandomForestClassifier": ensemble.RandomForestClassifier(random_state=RANDOM_STATE), # совсем плохие показатели
        #"ExtraTreeClassifier": tree.ExtraTreeClassifier(random_state=RANDOM_STATE), #
        "DecisionTreeClassifier": tree.DecisionTreeClassifier(random_state=RANDOM_STATE), # лучший по последним баллам
    }
    #return {**dictFast}
    #return {**dict_Test_MLPClassifier}
    #return {**dictFast, **dictLongTrain}
    return {**dictFastTree}




#import joblib
#REPO_ID = "YOUR_REPO_ID"
#FILENAME = "sklearn_model.joblib"
#model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))




if False:
    if False:
        # https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
        scaler = sklearn.preprocessing.StandardScaler()
        #scaler = sklearn.preprocessing.PowerTransformer()
        #scaler = sklearn.preprocessing.RobustScaler()
        #scaler = sklearn.preprocessing.MinMaxScaler() # https://scikit-learn.org/1.1/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
        #scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))
        #scaler = sklearn.preprocessing.QuantileTransformer()
        #scaler = sklearn.preprocessing.QuantileTransformer(output_distribution="normal")
        #scaler = sklearn.preprocessing.Normalizer() # всё на сферу кладёт - приводит к 1 длину вектора
        scale_columns = ["X","Y","Z"]
        scaledData = scaler.fit_transform(df3D[scale_columns])
        if False:
            scaler2 = sklearn.preprocessing.Normalizer()
            scaledData = scaler2.fit_transform(scaledData)
        df3D_Scaled = pd.DataFrame(data=scaledData, columns=scale_columns)
        df3D_Scaled["Vposneg"] = df3D["Vposneg"]
        df3D = df3D_Scaled
