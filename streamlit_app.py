import os, re, sys, time, math, shutil, urllib, string, random, pickle, zipfile, datetime
import streamlit as st, pandas as pd, numpy as np
import my_static_methods as my_stm

st.html(my_stm.STYLE_CORRECTION)
#st.sidebar.markdown("🧊 проверка по пакетам XYZ")
st.info("🧊 проверка предсказаний по пакетам XYZ")

def ReRun():
    try: st.rerun()
    except: pass

def DescriptionMarkdown() -> str:
    return """
        ## Описание
        ### 1) Загрузка нового файла
        Источником данных является файл CSV
    """

def save_dataframe_nodialog_idxyz(new_filename, dfToSave):
    commit_info = my_stm.save_dataframe_to_hf(REPO, dfToSave, new_filename, "ID_XYZ")
    st.toast(commit_info, icon='🆕')
    ReRun()


REPO = my_stm.HfRepo("f64k/gaziev", "dataset", st.secrets["HF_WRITE"])
lstRepoFiles = my_stm.list_files_hf(REPO)
dictTestFilesIdXyz = {f.upper().replace("ID_XYZ/",""): f.upper() for f in lstRepoFiles if f.upper().startswith("ID_XYZ/")}

col1, col2 = st.columns([1,1])
with col1.container():
    cont_cols = st.columns([1,2])
    cont_cols[0].popover("❓").markdown(DescriptionMarkdown())
    with cont_cols[1].popover("🆕 добавить новый файл"):
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
                    dgID = dfToUpload.groupby("ID")
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
    # список уже имеющихся файлов
    selectedFile = st.radio("📰 загруженные тестовые пакеты", dictTestFilesIdXyz.keys(), index=None)
    if selectedFile is not None:
        dict_ONE_IDXYZ = my_stm.load_dataframes_from_hf(REPO, [dictTestFilesIdXyz[selectedFile]])
        if len(dict_ONE_IDXYZ) > 0:
            df_idxyz = list(dict_ONE_IDXYZ.values())[0]
            dfShow = df_idxyz
            dgID = df_idxyz.groupby("ID")
            dictGroupID = dict(list(dgID))
            dfShow = dgID.apply(len) #.reset_index()
            col2.dataframe(dfShow, height=700)




