import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import os

# --- تنظیمات اولیه ---
st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠", layout="wide")

# --- مدیریت State ---
if 'selected_sample_path' not in st.session_state:
    st.session_state.selected_sample_path = None

# --- تابع لود کردن فونت لوکال ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

font_base64 = get_base64_of_bin_file("IRANSansDN.woff")

# تعریف CSS
if font_base64:
    # فونت فقط روی متون اعمال شود، نه روی آیکون‌ها
    custom_font_css = f"""
        @font-face {{
            font-family: 'IRANSans';
            src: url('data:font/woff;base64,{font_base64}') format('woff');
        }}
        /* اعمال فونت روی تگ‌های متنی اصلی */
        html, body, p, h1, h2, h3, h4, h5, h6, span, div, button, input, a, label {{ 
            font-family: 'IRANSans', 'Tahoma', sans-serif !important; 
        }}
    """
else:
    custom_font_css = "* { font-family: 'Tahoma', sans-serif; }"

# --- اعمال استایل CSS ---
st.markdown(f"""
    <style>
    {custom_font_css}
    
    /* تنظیمات کلی و راست‌چین */
    .main, .stMarkdown, .stButton, div {{direction: rtl; text-align: right;}}
    section[data-testid="stSidebar"] {{direction: rtl; text-align: right;}}
    h1, h2, h3 {{text-align: center; color: #2c3e50;}}
    .stAlert {{direction: rtl; text-align: right;}}
    
    /* بازگرداندن فونت آیکون‌ها به حالت استاندارد برای جلوگیری از نمایش متن keyboard_arrow */
    [data-testid="stIconMaterial"] {{
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        direction: ltr !important;
    }}
    
    /* حذف دکمه/فاصله مزاحم بالای سایدبار */
    section[data-testid="stSidebar"] div:first-child {{padding-top: 0rem;}}
    div.block-container {{padding-top: 2rem;}}
    
    /* استایل دکمه‌ها و تصاویر */
    .stLinkButton {{direction: rtl;}}
    div[data-testid="stImage"] {{display: block; margin-left: auto; margin-right: auto;}}
    
    /* مخفی کردن هدر دیفالت استریم‌لیت */
    header {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

# --- بارگذاری مدل ---
@st.cache_resource
def load_classification_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        return model
    except OSError:
        return None

try:
    model = load_classification_model()
except Exception as e:
    st.error(f"خطا در بارگذاری مدل: {e}")
    model = None

CLASS_NAMES = ['Glioma (گلیوما)', 'Meningioma (مننژیوم)', 'No Tumor (سالم)', 'Pituitary (هیپوفیز)']

def real_prediction(image, model):
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] != 3:
        img_array = np.stack((img_array,)*3, axis=-1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = 100 * np.max(predictions[0])
    
    return CLASS_NAMES[class_index], confidence, predictions[0]

# --- سایدبار ---
with st.sidebar:
    # لوگوی لوکال
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)

    st.title("پنل مشخصات")
    
    st.info("**درس:** چند رسانه ای (کارشناسی ارشد)")
    
    st.error("**استاد راهنما:** جناب آقای دکتر مهدی اسلامی")
    
    
    st.warning("**دانشجو:** اشکان حاجی بنده")
    
 
    st.info("**شماره دانشجویی:**\n\n403144020")
    
    
    st.markdown("---")
    st.markdown("### 📄 مقاله مرجع")
    
    st.markdown("**عنوان:** Brain Tumor Detection Based on Deep Learning Approaches and MRI")
    
    # باکس جداگانه ژورنال
    st.caption("**ژورنال:**")
    st.write("Cancers (2023)")
    
    # باکس جداگانه Impact Factor
    st.caption("**Impact Factor:**")
    st.write("5.2")
    
    st.markdown("---")
    st.link_button("🔗 مشاهده مقاله (PMC)", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10216636/")
    st.link_button("📊 دانلود دیتاست (Kaggle)", "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    
    st.success("وضعیت مدل: " + ("✅ آماده" if model else "❌ یافت نشد"))

# --- بدنه اصلی ---
st.title("سیستم تشخیص تومور مغزی با هوش مصنوعی")
st.markdown("<h5 style='text-align: center; color: #5d6d7e;'>تشخیص بیماری‌های گلیوما، مننژیوم، هیپوفیز و بافت سالم</h5>", unsafe_allow_html=True)
st.markdown("---")

final_image = None
image_source = st.radio("انتخاب منبع تصویر:", ("آپلود تصویر جدید", "استفاده از نمونه‌های تست"), horizontal=True)

if image_source == "آپلود تصویر جدید":
    st.session_state.selected_sample_path = None
    uploaded_file = st.file_uploader("تصویر MRI را آپلود کنید...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        final_image = Image.open(uploaded_file)

else:
    st.write("یک نمونه را جهت تست انتخاب کنید:")
    
    sample_dict = {
        "glioma": "samples/glioma.jpg",
        "meningioma": "samples/meningioma.jpg",
        "pituitary": "samples/pituitary.jpg",
        "notumor": "samples/notumor.jpg"
    }
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.write("Gilioma")
        if st.button("تست گلیوما"):
            st.session_state.selected_sample_path = sample_dict["glioma"]
            
    with col_s2:
        st.write("Meningioma")
        if st.button("تست مننژیوم"):
            st.session_state.selected_sample_path = sample_dict["meningioma"]
            
    with col_s3:
        st.write("Pituitary")
        if st.button("تست هیپوفیز"):
            st.session_state.selected_sample_path = sample_dict["pituitary"]
            
    with col_s4:
        st.write("No Tumor")
        if st.button("تست سالم"):
            st.session_state.selected_sample_path = sample_dict["notumor"]

    if st.session_state.selected_sample_path:
        if os.path.exists(st.session_state.selected_sample_path):
            final_image = Image.open(st.session_state.selected_sample_path)
        else:
            st.warning(f"فایل {st.session_state.selected_sample_path} یافت نشد.")

# --- نمایش و پردازش ---
if final_image is not None:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("تصویر ورودی:")
        st.image(final_image, use_container_width=True)
    
    with col2:
        st.success("نتایج آنالیز:")
        
        if model:
            if st.button("🔍 اجرای هوش مصنوعی", type="primary"):
                with st.spinner('در حال پردازش...'):
                    label, conf, all_probs = real_prediction(final_image, model)
                
                st.markdown(f"### نتیجه نهایی: **{label}**")
                
                if conf > 80: bar_color = "green"
                elif conf > 50: bar_color = "orange"
                else: bar_color = "red"
                
                st.progress(int(conf))
                st.caption(f"میزان اطمینان مدل: {conf:.2f}%")
                
                # اینجا مشکل فونت آیکون حل شده است
                with st.expander("مشاهده جزئیات احتمالات"):
                    for i, class_name in enumerate(CLASS_NAMES):
                        prob_val = all_probs[i] * 100
                        st.write(f"**{class_name}:** {prob_val:.2f}%")
                        st.progress(int(prob_val))
                    
        else:
            st.error("مدل بارگذاری نشده است.")
elif image_source == "استفاده از نمونه‌های تست" and st.session_state.selected_sample_path is None:
    st.info("لطفاً یکی از دکمه‌های بالا را کلیک کنید.")
