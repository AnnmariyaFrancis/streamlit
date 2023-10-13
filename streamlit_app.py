import streamlit as st
from PIL import Image

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#####################
# Header 
st.write('''
# Ann Mariya Francis.
##### *Resume* 
''')

image = Image.open('Ann_Mariya.jpg')
st.image(image, width=150)

st.markdown('## Summary', unsafe_allow_html=True)
st.info('''
- Dedicated data science specialist, recently graduated with a strong foundation in data analysis, machine learning, and statistical modeling.   
- Proficient in programming languages and excited to contribute to impactful projects while continuously expanding my skill set.  

''')

#####################
# Navigation

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #16A2CB;">
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item">
        <a class="nav-link" href="#education">Education</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#work-experience">Work Experience</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#social-media">Social Media</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)

#####################
# Custom function for printing text
def txt(a, b):
  col1, col2 = st.columns([4,1])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)

def txt2(a, b):
  col1, col2 = st.columns([1,4])
  with col1:
    st.markdown(f'`{a}`')
  with col2:
    st.markdown(b)

def txt3(a, b):
  col1, col2 = st.columns([1,2])
  with col1:
    st.markdown(a)
  with col2:
    st.markdown(b)
  
def txt4(a, b, c):
  col1, col2, col3 = st.columns([1.5,2,2])
  with col1:
    st.markdown(f'`{a}`')
  with col2:
    st.markdown(b)
  with col3:
    st.markdown(c)

#####################
st.markdown('''
## Education
''')

txt('**Master Of Science** (Data Analytics), *St. Joseph’s College*,Irinjalakuda',
'2021-2023')
st.markdown('''
- Published paper "Data Science in Education In this paper, we discuss how data science can be applied to education "  
- Participated in regional-level ideathon conducted by IEDC   
- A national-level one-day workshop on "Cloud Technologies And The Role Of Cloud In Data 
  Science" arranged by  St. Joseph's College, Irinjalakuda     
  
''')

txt('**Bachelor of Science** (Mathematics), *St. Joseph’s College*,Irinjalakuda','2018-2021')
st.markdown('''
- Participated in National Integration Camp for NCC Cadets
- Graduated with First Class .
''')

#####################
st.markdown('''
## Work Experience
''')

txt('**Data Analyst intern**,Yoshops,Chennai,India','Feb 2023 -April 2023')
st.markdown('''
- Implemented a more sophisticated sales forecasting model at Yoshops, resulting in a 15% increase in forecast accuracy.
- Designed and implemented a user-friendly data visualization dashboard using PowerBI.This dashboard reduced the time spent on data analysis by 30%.
''')

txt('**Data Science intern**, Luminar Techno lab ,Ernakulam,kerala,India',
'Jan 2023-July 2023')
st.markdown('''
- Spearheaded data preprocessing, model development, and optimization efforts, resulting in a 40% reduction in data processing time and a 25% improvement in model accuracy.
''')

txt('**TCS iON Remote internship**','June 2023-Aug 2023')
st.markdown('''
- Worked on the project "•	Automate Extraction of Handwritten Text from an Image "
''')
#####################
st.markdown('''
## Skills
''')
txt3('Programming', '`Python`, `R`, `Linux`')
txt3('Data processing/wrangling', '`SQL`, `pandas`, `numpy`')
txt3('Data visualization', '`matplotlib`, `seaborn`, `plotly`, `ggplot2`,`PowerBi`,`Tableau`')
txt3('Machine Learning', '`scikit-learn`')
txt3('Deep Learning', '`TensorFlow`,`PyTorch`,`XGBoost`')
txt3('Web development', '`Flask`, `HTML`, `CSS`')
txt3('Model deployment', '`streamlit`')

#####################
st.markdown('''
## Social Media
''')
txt2('LinkedIn', 'https://www.linkedin.com/in/ann-mariya-francis-3b29b3216?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BjZ%2FJvUghS86XIr9n2JqFkg%3D%3D')

txt2('GitHub', 'https://github.com/AnnmariyaFrancis')
txt2('Leetcode', 'https://leetcode.com/annmariyafrancis1631/')

