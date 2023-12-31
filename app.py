import streamlit as st
import numpy as np
import joblib

data = joblib.load('copper_reg_model.joblib')

def predict_selling_price(fns_input):
    input_array = np.array(fns_input).reshape(1, -1)
    prediction=(data.predict(input_array))**2
    prediction = np.round(*prediction)
    return prediction

def main():
    result = None
    status_dict = {'Won': 116012, 'Lost': 34438, 'Not lost for AM': 19573, 'Revised': 4276, 'To be approved': 4170, 
              'Draft': 3140, 'Offered': 53, 'Offerable': 10, 'Wonderful': 1}
    item_type_dict = {'W': 105615, 'S': 69236, 'PL': 5660, 'Others': 610, 'WI': 524, 'IPL': 27, 'SLAWR': 1}
    country_list = [ 28.,  25.,  30.,  32.,  38.,  78.,  27.,  77., 113.,  79.,  26., 39.,  40.,  84.,  80., 107.,  89.]
    application_list = [10., 41., 28., 59., 15.,  4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79.,
                  3., 99.,  2.,  5., 39., 69., 70., 65., 58., 68.]
    product_ref_list = [1670798778, 1668701718,     628377,     640665,     611993, 1668701376,  164141591, 1671863738, 1332077137, 640405,
                 1693867550, 1665572374, 1282007633, 1668701698,     628117, 1690738206,     628112,     640400, 1671876026,
                 164336407, 164337175, 1668701725, 1665572032,     611728, 1721130331, 1693867563, 611733, 1690738219, 
                 1722207579,  929423819, 1665584320, 1665584662, 1665584642]

    st.set_page_config(layout="wide")

    st.write("""
    <div style='text-align:center'>
        <h1 style='color:#009999;'>Industrial Copper Selling Price Prediction Application</h1>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("my_form"):
        custom_css = """
        <style>
            .stNumberInput input {
                font-size: 20px; /* Adjust the font size to your preference */
            }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)    
        st.markdown('<h4 style="font-size: 20px;">Status</h4>', unsafe_allow_html=True)
        selected_status = st.selectbox("", list(status_dict.keys()), key=1)
        st.markdown('<h4 style="font-size: 20px;">Item Type</h4>', unsafe_allow_html=True)
        selected_item_type = st.selectbox("", list(item_type_dict.keys()), key=2)
        st.markdown('<h4 style="font-size: 20px;">Country</h4>', unsafe_allow_html=True)
        selected_country = st.selectbox("", sorted(country_list), key=3)
        st.markdown('<h4 style="font-size: 20px;">Application</h4>', unsafe_allow_html=True)
        selected_application = st.selectbox("", sorted(application_list), key=4)
        st.markdown('<h4 style="font-size: 20px;">Product Reference</h4>', unsafe_allow_html=True)
        selected_product_ref = st.selectbox("", sorted(product_ref_list), key=5)
        st.write(f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value between min and max</h5>', unsafe_allow_html=True)
        st.markdown('<h4 style="font-size: 20px;">Enter Quantity (in Tons)(Min:1 and Max:70253)</h4>', unsafe_allow_html=True)
        selected_quantity_tons = st.number_input('',min_value=1, max_value=70253)
        st.markdown('<h4 style="font-size: 20px;">Enter customer ID (Min:12458 and Max:2147484000)</h4>', unsafe_allow_html=True)
        selected_customer = st.number_input("", min_value=12458, max_value=2147484000)
        st.markdown('<h4 style="font-size: 20px;">Enter thickness (Min:0.18 and Max:25.5)</h4>', unsafe_allow_html=True)
        selected_thickness = st.number_input("", min_value=0.18, max_value=25.5)
        st.markdown('<h4 style="font-size: 20px;">Enter width (Min:1 and Max:2990)</h4>', unsafe_allow_html=True)
        selected_width = st.number_input("", min_value=1, max_value=2990)
        st.markdown('<h4 style="font-size: 20px;">Enter date (Min:1 and Max:31)</h4>', unsafe_allow_html=True)
        selected_item_date = st.number_input("", min_value=1, max_value=31)
        st.markdown('<h4 style="font-size: 20px;">Enter month (Min:1 and Max:12)</h4>', unsafe_allow_html=True)
        selected_item_month = st.number_input("", min_value=1, max_value=12)
        st.markdown('<h4 style="font-size: 20px;">Enter year (Min:1995 and Max:2021)</h4>', unsafe_allow_html=True)
        selected_item_year = st.number_input("", min_value=1995, max_value=2021)
        submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: #009999;
                color: white;
                width: 100%;
            }
            </style>
        """, unsafe_allow_html=True)

    if submit_button:
        quantity_tons = np.log1p(float(selected_quantity_tons))
        customer = float(selected_customer)
        country = float(selected_country)
        status = int(status_dict[selected_status])
        item_type = int(item_type_dict[selected_item_type])
        application = float(selected_application)
        thickness = np.log1p(float((selected_thickness)))
        width = float(selected_width)
        product_ref = int(selected_product_ref)
        item_delivery_year = int(selected_item_year)
        item_delivery_month = int(selected_item_month)
        item_delivery_date = int(selected_item_date)
        fns_input = [customer, country, status, item_type, application, thickness, width, product_ref,
                     quantity_tons, item_delivery_year, item_delivery_month, item_delivery_date]
        result = predict_selling_price(fns_input)  
    st.markdown(f'<div style="color: green; font-size: 30px;">The predicted selling price is: Rs.{result}</div>', unsafe_allow_html=True)    
    
if __name__=='__main__':
    main()