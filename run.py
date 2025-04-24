# streamlit_app.py
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
     page_icon="🌾",
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
 
 
 # تحميل صورة بانر
 
 # الشريط الجانبي لاختيار طريقة الإدخال
 logo = Image.open('MOE_logo.png')
 st.sidebar.image(logo,  width=170)
 st.sidebar.header("خيارات الإدخال")
 input_method = st.sidebar.radio("اختر طريقة الإدخال:", ("إدخال يدوي", "تحميل ملف (CSV)"))
 
 # تعريف الأعمدة المطلوبة
 required_columns = [
     'activity_total_area_hectares',
     'well_irrigation_type_1.0',
     'wells_number',
     'well_possession_type_1',
     'well_is_active_1',
     'activity_irrigation_type_1.0',
     'activity_type_2.0',
     'property_area',
     'farm_main_crop_type_1.0',
     'well_irrigation_source_2',
     'activity_unique_crop_types_count',
     'activity_productive_trees_count',
     'activity_irrigation_source_2.0',
     'well_irrigation_source_1',
     'activity_status_1',
     'activity_protected_house_type_1.0',
     'activity_count',
     'activity_irrigation_type_2.0',
 ]
 
 # دالة لتنفيذ التوقعات
 def perform_prediction(input_data):
     # إجراء الحسابات
     input_data['sprinklers_count'] = input_data['wells_number']
     input_data['sprinklers_count_kw'] = 25 * input_data['wells_number']
 
     # حساب النسب المئوية
     input_data['well_irrigation_type_1.0_percentage'] = input_data.apply(
         lambda row: row['well_irrigation_type_1.0'] / row['wells_number'] if row['wells_number'] != 0 else 0,
         axis=1
     )
     # افتراض 'well_irrigation_type_3.0_percentage' كـ 0 إذا لم يتم توفيرها
     input_data['well_irrigation_type_3.0_percentage'] = 0
 
     # التعامل مع القسمة على الصفر
     def safe_divide(a, b):
         return a / b if b != 0 else 0
 
     # إنشاء ميزات جديدة حسب إعداد البيانات
     input_data['trees_per_hectare'] = input_data.apply(
         lambda row: safe_divide(row['activity_productive_trees_count'], row['activity_total_area_hectares']),
         axis=1
     )
     input_data['irrigation_intensity'] = (
         input_data['activity_irrigation_type_2.0'] * input_data['activity_total_area_hectares']
     )
     input_data['well_density'] = input_data.apply(
         lambda row: safe_divide(row['wells_number'], row['activity_total_area_hectares']),
         axis=1
     )
     input_data['area_per_activity'] = input_data.apply(
         lambda row: safe_divide(row['activity_total_area_hectares'], row['activity_count']),
         axis=1
     )
 
     # استبدال القيم اللانهائية و NaN الناتجة عن القسمة على الصفر
     input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
     input_data.fillna(0, inplace=True)
 
     # تحضير البيانات النهائية للتوقع
     final_input_data = input_data[
         [
             'activity_total_area_hectares',
             'well_irrigation_type_1.0',
             'sprinklers_count_kw',
             'sprinklers_count',
             'wells_number',
             'well_possession_type_1',
             'well_is_active_1',
             'activity_irrigation_type_1.0',
             'activity_type_2.0',
             'property_area',
             'farm_main_crop_type_1.0',
             'well_irrigation_source_2',
             'activity_unique_crop_types_count',
             'activity_productive_trees_count',
             'well_irrigation_type_1.0_percentage',
             'activity_irrigation_source_2.0',
             'well_irrigation_source_1',
             'well_irrigation_type_3.0_percentage',
             'activity_status_1',
             'activity_protected_house_type_1.0',
             'activity_count',
             'activity_irrigation_type_2.0',
             'trees_per_hectare',
             'irrigation_intensity',
             'well_density',
             'area_per_activity'
         ]
     ]
     st.write(final_input_data)
 
     classifier_prediction = stacking_classifier.predict(final_input_data)
     area_class = label_encoder.inverse_transform(classifier_prediction)
     final_input_data['area_class'] = area_class
     st.write(final_input_data['area_class'])
 
     # توقع الانحدار
     prediction_log = stacking_regressor.predict(final_input_data)
     prediction = np.expm1(prediction_log)
 
     if int(final_input_data.iloc[0]['activity_total_area_hectares']) == 0 or int(final_input_data.iloc[0]['property_area']) == 0 or int(final_input_data.iloc[0]['activity_count']) == 0 or int(final_input_data.iloc[0]['wells_number']) == 0:
         return 0, final_input_data
 
     return prediction[0], final_input_data
 
 if input_method == "إدخال يدوي":
 
     # استخدام علامات تبويب لتنظيم أقسام الإدخال
     tabs = st.tabs(["تفاصيل المزرعة", "تفاصيل الآبار", "تفاصيل المحصول"])
 
     with tabs[0]:
         st.subheader("تفاصيل المزرعة والنشاط")
         cols = st.columns(2)
         with cols[0]:
             property_area = st.number_input(
                 'مساحة المزرعة (هكتار)',
                 min_value=0.0, step=0.0001, format="%.2f",
                 help='المساحة الإجمالية للمزرعة بالهكتار.'
             )
             activity_total_area_hectares = st.number_input(
                 'إجمالي مساحة النشاطات في المزرعة (هكتار)',
                 min_value=0.0, step=0.0001, format="%.2f",
                 help='المساحة الإجمالية للنشاطات في المزرعة بالهكتار.'
             )
             activity_count = st.number_input(
                 'عدد أنشطة المزرعة',
                 min_value=1, step=1, value=1,
                 help='إجمالي عدد الأنشطة في المزرعة.'
             )
 
             activity_productive_trees_count = st.number_input(
                 'عدد الأشجار ',
                 min_value=0, step=1, value=0,
                 help='إجمالي عدد الأشجار في المزرعة.'
             )
 
             activity_irrigation_type_1_0 = st.number_input(
                 'عدد أنشطة الري المحوري',
                 min_value=0, step=1, value=0,
                 help='عدد نشاطات الري المحوري أو المعروف في حصر ب1.0.'
             )
         with cols[1]:
             activity_irrigation_type_2_0 = st.number_input(
                 'عدد أنشطة الري بالغمر',
                 min_value=0, step=1, value=0,
                 help='عدد نشاطات الري بالغمر أو المعروف في حصر ب2.0.'
             )
             activity_irrigation_source_2_0 = st.number_input(
                 'عدد الأنشطة التي تعتمد على مصادر البئر الإرتوازي',
                 min_value=0, step=1, value=0,
                 help='عدد مصدر الري 2.0 في النشاط.'
             )
             activity_type_2_0 = st.number_input(
                 'عدد الأنشطة المصنفة كري المحوري',
                 min_value=0, step=1, value=0,
                 help='عدد نوع النشاط 2.0.'
             )
             activity_status_1 = st.number_input(
                 'عدد الأنشطة القائمة',
                 min_value=0, step=1, value=0,
                 help='عدد حالة النشاط 1.'
             )
             activity_protected_house_type_1_0 = st.number_input(
                 'عدد البيوت المحمية البلاستيكية العادية',
                 min_value=0, step=1, value=0,
                 help='عدد نوع البيت المحمي 1.0 في النشاط.'
             )
 
     with tabs[1]:
         st.subheader("تفاصيل الآبار")
         wells_number = st.number_input(
             'عدد الآبار',
             min_value=0, step=1, value=1,
             help='إجمالي عدد الآبار في المزرعة.'
         )
         well_irrigation_type_1_0 = st.number_input(
             'عدد الآبار التي تخدم الري المحوري',
             min_value=0, step=1, value=0,
             help='عدد الآبار ذات نوع الري 1.0.'
         )
         well_possession_type_1 = st.number_input(
             'عدد الآبار المملوكة',
             min_value=0, step=1, value=0,
             help='عدد الآبار ذات نوع الحيازة 1.'
         )
         well_is_active_1 = st.number_input(
             'عدد الآبار النشطة',
             min_value=0, step=1, value=0,
             help='عدد الآبار النشطة.'
         )
         well_irrigation_source_1 = st.number_input(
             'عدد الآبار من نوع بئر عادي',
             min_value=0, step=1, value=0,
             help='عدد الآبار ذات مصدر الري 1.'
         )
         well_irrigation_source_2 = st.number_input(
             'عدد الآبار من نوع بئر ارتوازي',
             min_value=0, step=1, value=0,
             help='عدد الآبار ذات مصدر الري 2.'
         )
 
     with tabs[2]:
         st.subheader("تفاصيل المحصول")
         farm_main_crop_type_1_0 = st.number_input(
             'عدد الأنشطة المنتجة للأعلاف',
             min_value=0, step=1, value=0,
             help='عدد نوع المحصول الرئيسي 1.0 في المزرعة.'
         )
         activity_unique_crop_types_count = st.number_input(
                 'عدد المحاصيل المختلفة المنتجة في المزرعة',
                 min_value=0, step=1, value=1,
                 help=' أنواع المحاصيل المختلفة المنتجة في المزرعة.'
             )
 
     # التوقع للإدخال اليدوي
     bt_cols = st.columns(5)
     with bt_cols[2]:
         if st.button('توقع', key='manual_predict'):
             with st.spinner('جارٍ الحساب...'):
                 # إنشاء DataFrame بالبيانات المدخلة
                 input_data = pd.DataFrame({
                     'activity_total_area_hectares': [activity_total_area_hectares],
                     'well_irrigation_type_1.0': [well_irrigation_type_1_0],
                     'wells_number': [wells_number],
                     'well_possession_type_1': [well_possession_type_1],
                     'well_is_active_1': [well_is_active_1],
                     'activity_irrigation_type_1.0': [activity_irrigation_type_1_0],
                     'activity_type_2.0': [activity_type_2_0],
                     'property_area': [property_area*10_000],
                     'farm_main_crop_type_1.0': [farm_main_crop_type_1_0],
                     'well_irrigation_source_2': [well_irrigation_source_2],
                     'activity_unique_crop_types_count': [activity_unique_crop_types_count],
                     'activity_productive_trees_count': [activity_productive_trees_count],
                     'activity_irrigation_source_2.0': [activity_irrigation_source_2_0],
                     'well_irrigation_source_1': [well_irrigation_source_1],
                     'activity_status_1': [activity_status_1],
                     'activity_protected_house_type_1.0': [activity_protected_house_type_1_0],
                     'activity_count': [activity_count],
                     'activity_irrigation_type_2.0': [activity_irrigation_type_2_0],
                 })
 
                 # تنفيذ التوقع
                 prediction, final_input_data = perform_prediction(input_data)
 
                 st.success(f'إجمالي الحمل الكهربائي المقدر (كيلوواط): **{prediction:.2f}**')
 
 
 else:
     st.header("رفع ملف البيانات بصيغة (CSV)")
     st.write("يرجى التأكد من أن الملف يحتوي على البيانات التالية وبالصيغة المطلوبة")
     st.code(", ".join(required_columns), language='plaintext')
 
     # خيار تنزيل ملف CSV نموذجي
     @st.cache_data
     def convert_df(df):
         return df.to_csv(index=False).encode('utf-8')
 
     sample_df = pd.DataFrame(columns=required_columns)
     csv_sample = convert_df(sample_df)
     st.download_button(
         label="📥 تنزيل ملف CSV نموذجي",
         data=csv_sample,
         file_name='sample_input.csv',
         mime='text/csv',
     )
 
     uploaded_file = st.file_uploader("اختر ملف CSV", type=["csv"], accept_multiple_files=False)
 
     if uploaded_file is not None:
         try:
             # قراءة ملف CSV
             data = pd.read_csv(uploaded_file)
             st.write(f"البيانات المحملة تحتوي على {data.shape[0]} صفوف و {data.shape[1]} أعمدة.")
 
             # التحقق من الأعمدة المطلوبة المفقودة
             missing_columns = [col for col in required_columns if col not in data.columns]
             if missing_columns:
                 st.error(f"الأعمدة المطلوبة التالية مفقودة في ملف CSV المحمل: {missing_columns}")
             else:
                 if st.button('توقع', key='csv_predict'):
                     with st.spinner('جارٍ الحساب...'):
                         # تنفيذ التوقع
                         predictions = []
                         for idx, row in data.iterrows():
                             input_data = pd.DataFrame([row])
                             prediction, _ = perform_prediction(input_data)
                             predictions.append(prediction)
 
                         data['إجمالي الحمل الكهربائي المقدر (كيلوواط)'] = predictions
                        # إضافة عمود الديزل
                        data['إجمالي استهلاك الديزل المكافئ'] = data['إجمالي الحمل الكهربائي المقدر (كيلوواط)'] * 3.5

                        st.success('تم إكمال التوقعات!')
                        st.subheader('التوقعات:')
                        st.dataframe(
                            data[[
                                'إجمالي الحمل الكهربائي المقدر (كيلوواط)',
                                'إجمالي استهلاك الديزل المكافئ'
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
 
         except Exception as e:
             st.error(f"حدث خطأ: {e}")
     else:
         st.info("للمتابعة يرجى رفع ملف البيانات")
