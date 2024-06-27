import os, re, sys, time, math, shutil, urllib, string, random, pickle, zipfile, datetime
import streamlit as st, pandas as pd, numpy as np
import my_static_methods as my_stm

st.html(my_stm.STYLE_CORRECTION)

REPO = my_stm.HfRepo("f64k/gaziev", "dataset", st.secrets["HF_WRITE"])
lstRepoFiles = my_stm.list_files_hf(REPO) # список уже имеющихся в репозитории файлов
dictTestFilesIdXyz = {f.upper().replace("ID_XYZ/",""): f.upper() for f in lstRepoFiles if f.upper().startswith("ID_XYZ/")}

@st.cache_data
def GetListOf_XYZV_ToTrainClassifier(repo):
    lstRepoZipFiles = ["TrainData_1504_AB_gaziev.zip","TestData_1504_AB_gaziev.zip","TestData3_2204_noAB_gaziev.zip"]
    dictTrainThreeDataframes = my_stm.load_dataframes_from_hf(REPO, lstRepoZipFiles)
    lstDfOriginal = [my_stm.df_process_v_column(df) for df in dictTrainThreeDataframes.values()]
    return lstDfOriginal

def ReRun() :
    try: st.rerun()
    except: pass

def DescriptionMarkdown() -> str:
    return """
        ## Описание
        ### 1) Загрузка нового файла
        Источником данных является файл CSV. Первая строка - названия столбцов ID;X;Y;Z\n
        ### Данные для фактической работы (прогнозирования)
        Аналогично данным для обучения это строки цифр (разделенных «;»), но содержащих только рабочие параметры.\n
        Количество строк должно соответствовать настройкам конкретной модели, при этом сервис обязан контролировать структуру этих данных, но не осуществляет их подготовку.\n
        Данные для работы также представляются во внешнем файле .csv \n
        При этом необходима возможность организации в одном файле до 500 «пакетов» (строк цифр необходимых для единоразовой отработки обученной моделью) данных, а также специальный уникальный идентификатор каждого пакета.\n
        Соответственно, будет происходить последовательная обработка каждого пакета (задания).\n
        Формат данных пакетного файла и идентификатора – csv.\n
        Обработка данных происходит до нескольких раз в день.\n
        Время обработки каждого пакета (задания) не более 1 минуты\n
    """

def save_dataframe_nodialog_idxyz(new_filename, dfToSave):
    commit_info = my_stm.save_dataframe_to_hf(REPO, dfToSave, new_filename, "ID_XYZ")
    st.toast(commit_info, icon='🆕')
    ReRun()

#st.sidebar.markdown("🧊 проверка по пакетам XYZ")

with st.container():
    cols1 = st.columns([1,12]) # vertical_alignment: "center"
    cols1[0].popover("❓", help="пояснения").markdown(DescriptionMarkdown())
    cols1[1].info("🔮 проверка предсказаний по пакетам ID_XYZ. 📜 формат CSV. 🧊 названия столбцов ID;X;Y;Z. 📐 размер пакетов одинаковый.")

#col1, col2 = st.columns([2,5])
col1, col2 = st.columns([4,2])

with col1.popover("🆕 добавить новый файл", use_container_width=False):
    uploaded_file = st.file_uploader("💾 “откройте CSV для загрузки”", ["csv"])
    if uploaded_file is not None:
        dfLoaded = None
        delim = ";"
        enc = "utf-8"
        if uploaded_file.type == "text/csv":
            try: dfLoaded = pd.read_csv(uploaded_file, sep=delim, encoding=enc)
            except Exception as ex: st.write(ex)
        else:
            if uploaded_file.type == "application/x-zip-compressed":
                try: dfLoaded = pd.read_csv(uploaded_file, sep=delim, encoding=enc, compression="zip")
                except Exception as ex: st.write(ex)
            else:
                st.error(uploaded_file.type)
        # dataframe ready. try to upload to HF
        if not dfLoaded is None:
            dfToUpload = dfLoaded
            if "ID" in dfToUpload.columns:
                dfToUpload = dfLoaded.query("ID!='ID'")
            #col2.dataframe(df)
            colnames = "".join(dfToUpload.columns)
            if colnames.upper().startswith("IDXYZ"):
                dgID = dfToUpload.groupby("ID") # , include_groups=False
                dictGroupID = dict(list(dgID))
                lstGroupIDs = list(dictGroupID.keys())
                #col2.write(dictGroupID)
                lst_len = list(set(dgID.apply(len)))
                if len(lst_len) == 1:
                    fileXYZ = f"{colnames}_{len(dictGroupID)}x{lst_len[0]}_{lstGroupIDs[0]}_{lstGroupIDs[-1]}.csv".upper()
                    if fileXYZ in dictTestFilesIdXyz.keys():
                        if st.button(f"такой файл есть! перезаписать файл '{fileXYZ}'?"):
                            save_dataframe_nodialog_idxyz(fileXYZ, dfToUpload)
                    else:
                        save_dataframe_nodialog_idxyz(fileXYZ, dfToUpload)
                else:
                    st.error(f"Разные размеры пакетов для разных ID, варианты : {lst_len}")
            else:
                st.error(f"Столбцы не ID;X;Y;Z ! Наблюдаем столбцы : {colnames}")

# список уже имеющихся в репозитории файлов. повторное чтение
lstRepoFiles = my_stm.list_files_hf(REPO)
dictTestFilesIdXyz = {f.upper().replace("ID_XYZ/",""): f.upper() for f in lstRepoFiles if f.upper().startswith("ID_XYZ/")}
selectedFile = col1.radio("📰 загруженные тестовые пакеты", dictTestFilesIdXyz.keys(), index=None)

# выбран файл для предсказания
if selectedFile is not None:
    dict_ONE_IDXYZ = my_stm.load_dataframes_from_hf(REPO, [dictTestFilesIdXyz[selectedFile]])
    if len(dict_ONE_IDXYZ) > 0:
        df_idxyz = list(dict_ONE_IDXYZ.values())[0]
        dfShow = df_idxyz
        dgID = df_idxyz.groupby("ID") # , include_groups=False
        dictGroupID = dict(list(dgID))
        dfShow = dgID.apply(len).reset_index()
        #col1.dataframe(dfShow, height=400)
        pack_size = list(set(dgID.apply(len)))[0]
        lstDfOriginal = GetListOf_XYZV_ToTrainClassifier(REPO)
        classifier_object, df_train_with_predict, time_elapsed = my_stm.GetClassifier(lstDfOriginal, pack_size-1)
        col2.popover(type(classifier_object).__name__).write(type(classifier_object))
        # прогноз на обучающей выборке
        columns_xyzv = [c for c in df_train_with_predict.columns if "Vis" in c] + [c for c in df_train_with_predict.columns if c[0] in "XYZ"]
        #col2.dataframe(df_train_with_predict[columns_xyzv], height=650)
        # расчет пакетов
        xyz = ["X","Y","Z"]
        df_packs_reshaped = dgID.apply(lambda df: pd.Series(df[xyz].values[::-1].reshape(1,-1)[0])).reset_index()
        x_test_vect = df_packs_reshaped.iloc[:,1:]
        df_packs_reshaped["Прогноз_V"] = classifier_object.predict(x_test_vect.values)
        col2.dataframe(df_packs_reshaped[["ID","Прогноз_V"]], height=620)
        # для отладки
        # col2.write(st.session_state)


