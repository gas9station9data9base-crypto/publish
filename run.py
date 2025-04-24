from sklearn.base import BaseEstimator, TransformerMixin
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=0.01, upper_percentile=0.99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.lower_percentile * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile * 100, axis=0)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_clipped

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, factor=1.5):
        self.variables = variables
        self.factor = factor

    def fit(self, X, y=None):
        self.cap_dict_ = {}
        for var in self.variables:
            Q1 = X[var].quantile(0.25)
            Q3 = X[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_cap = Q1 - self.factor * IQR
            upper_cap = Q3 + self.factor * IQR
            self.cap_dict_[var] = (lower_cap, upper_cap)
        return self

    def transform(self, X):
        X_capped = X.copy()
        for var in self.variables:
            lower_cap, upper_cap = self.cap_dict_[var]
            X_capped[var] = np.where(
                X_capped[var] < lower_cap, lower_cap,
                np.where(X_capped[var] > upper_cap, upper_cap, X_capped[var])
            )
        return X_capped

# تحميل نماذج التدريب
stacking_regressor = joblib.load('stacking_regressor.joblib')
stacking_classifier = joblib.load('stacking_classifier.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# إعداد صفحة التطبيق لتحسين التخطيط
st.set_page_config(
    page_title="تقدير الأحمال الكهربائية في المزارع",
    page_icon="MOE_logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        color: #4CAF50;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #8C8C8C;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        color: #8C8C8C;
        text-align: center;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# عنوان التطبيق والوصف
st.markdown('<h1 class="title">تقدير الأحمال الكهربائية في المزارع</h1>', unsafe_allow_html=True)
st.markdown('<h4>يعتمد  البرنامج في تقدير أحمال المزرعة على بيانات حصر والزيارات التي تمت على المزارع</h4>', unsafe_allow_html=True)

# الشريط الجانبي لاختيار طريقة الإدخال
logo = Image.open('MOE_logo.png')
st.sidebar.image(logo,  width=170)
st.sidebar.header("خيارات الإدخال")
input_method = st.sidebar.radio("اختر طريقة الإدخال:", ("إدخال يدوي", "تحميل ملف (CSV)"))

# تعريف الأعمدة المطلوبة
required_columns = [
    # same as before
]

# دالة لتنفيذ التوقعات
# (unchanged perform_prediction)

def perform_prediction(input_data):
    # ... existing feature engineering ...
    # توقع الانحدار
    prediction_log = stacking_regressor.predict(final_input_data)
    prediction = np.expm1(prediction_log)

    if int(final_input_data.iloc[0]['activity_total_area_hectares']) == 0 or int(final_input_data.iloc[0]['property_area']) == 0 or int(final_input_data.iloc[0]['activity_count']) == 0 or int(final_input_data.iloc[0]['wells_number']) == 0:
        return 0, final_input_data

    return prediction[0], final_input_data

if input_method == "إدخال يدوي":
    # ... input fields as before ...
    bt_cols = st.columns(5)
    with bt_cols[2]:
        if st.button('توقع', key='manual_predict'):
            with st.spinner('جارٍ الحساب...'):
                # جمع البيانات وإنشاء DataFrame
                input_data = pd.DataFrame({
                    # fields as before
                })
                prediction, final_input_data = perform_prediction(input_data)

                # عرض النتائج بوحدتي kW والديزل
                st.success(f'إجمالي الحمل الكهربائي المقدر (كيلوواط): **{prediction:.2f}**')
                diesel = prediction * 3.5
                st.success(f'إجمالي استهلاك الديزل المكافئ: **{diesel:.2f}**')

else:
    # تحميل CSV
    # ... file uploader and validation as before ...
    if uploaded_file is not None:
        # ... error handling as before ...
                if st.button('توقع', key='csv_predict'):
                    with st.spinner('جارٍ الحساب...'):
                        predictions = []
                        for idx, row in data.iterrows():
                            input_data = pd.DataFrame([row])
                            pred, _ = perform_prediction(input_data)
                            predictions.append(pred)

                        data['إجمالي الحمل الكهربائي المقدر (كيلوواط)'] = predictions
                        # إضافة عمود الديزل
                        data['إجمالي استهلاك الديزل المكافئ'] = data['إجمالي الحمل الكهربائي المقدر (كيلوواط)'] * 0.35

                        st.success('تم إكمال التوقعات!')
                        st.subheader('التوقعات:')
                        st.dataframe(
                            data[[
                                'إجمالي الحمل الكهربائي المقدر (كيلوواط)',
                                ' إجمالي استهلاك الديزل المكافئ (معامل الضرب المستخدم 0.35)'
                            ]]
                        )

                        # خيار تنزيل التوقعات
                        csv = data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 تنزيل التوقعات كملف CSV",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv',
                        )
