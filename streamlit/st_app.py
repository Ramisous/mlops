from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import streamlit as st
from config import config
from tagifai import data, utils

projects_fp = Path(config.DATA_DIR, "projects.json")
tags_fp = Path(config.DATA_DIR, "tags.json")
projects = utils.load_dict(filepath=projects_fp)
# print(utils.load_dict(filepath=tags_fp))
tags_dict = utils.list_to_dict(utils.load_dict(filepath=tags_fp), key="tag")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Projects (sample)")
    st.write(projects[0])
with col2:
    st.subheader("Tag")
    tag = st.selectbox("Choose a tag", list(tags_dict.keys()))
    st.write(tags_dict[tag])

df = pd.DataFrame(projects)
st.text(f"Projects (count: {len(df)}):")
st.write(df)


# print(df.tag)
num_tags_per_project = [len(tags) for tags in df.tag]  # df is dependent on min_freq slider's value
# print('num_tags_per_project',num_tags_per_project)
# print('zip(*Counter(num_tags_per_project).items())' , Counter(num_tags_per_project).items())
num_tags, num_projects = zip(*Counter(num_tags_per_project).items())
plt.figure(figsize=(10, 3))
ax = sns.barplot(list(num_tags), list(num_projects))
plt.title("Tags per project", fontsize=20)
plt.xlabel("Number of tags", fontsize=16)
ax.set_xticklabels(range(1, len(num_tags) + 1), rotation=0, fontsize=16)
plt.ylabel("Number of projects", fontsize=16)
plt.show()
st.pyplot(plt)


filters = st.text_input("filters", "[!\"'#$%&()*+,-./:;<=>?@\\[]^_`{|}~]")
lower = st.checkbox("lower", True)
stem = st.checkbox("stem", False)
text = st.text_input("Input text", "Conditional generation using Variational Autoencoders.")
preprocessed_text = data.clean_text(text=text, lower=lower, stem=stem)
st.write("Preprocessed text", preprocessed_text)
