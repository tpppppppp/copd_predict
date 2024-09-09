import pandas as pd
import numpy as np
import streamlit as st
import shap
# from pycaret.classification import load_model, predict_model
import streamlit.components.v1 as components
import pickle


# 加载模型
# model_best = load_model('best_model')

# 提取 PyCaret 模型中的基础模型
# def extract_core_model(pycaret_model):
#     if hasattr(pycaret_model, 'steps'):
#         core_model = pycaret_model.steps[-1][1]  # 提取管道中的最后一个步骤模型
#     else:
#         core_model = pycaret_model
#     return core_model

# 使用 SHAP 解释器进行可视化
def draw_force(model_best, X):
    #     core_model = extract_core_model(model_best)
    #     explainer = shap.Explainer(core_model)
    explainer = shap.Explainer(model_best)
    shap_values = explainer(X)
    # print(explainer.expected_value)
    # 显示 force_plot 到 Streamlit
    shap_html = shap.force_plot(explainer.expected_value[0], shap_values.values[0, :, 1], X.iloc[0, :], link='logit', matplotlib=False)
    return shap_html


# 用于将 SHAP HTML 嵌入 Streamlit
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)


# 预测函数
# def prediction1(input_data):
#     predictions = predict_model(model_best, data=input_data)
#     return predictions


def main():
    with open('mymodel.pkl', 'rb') as f:
        model_best = pickle.load(f)

    st.title("Hospitalization death prediction")
    #     html_temp = """
    #     <div style="background-color: #FFFF00; padding: 16px">
    #     <h1 style="color: #000000; text-align: center;">COPD Prediction ML App</h1>
    #     </div>
    #     """
    #     st.markdown(html_temp, unsafe_allow_html=True)

    # 收集用户输入
    NEUT1 = st.text_input("NEUT%", "73")
    NTproBNP1 = st.text_input("NT-proBNP", "6496")
    cTnT1 = st.text_input("cTnT", "0.03")
    DD1 = st.text_input("DD", "1335")
    PAD1 = st.text_input("PAD", "22")
    RAD1 = st.text_input("RAD", "41")
    PV1 = st.text_input("PV", "85")
    Vmax1 = st.text_input("Vmax", "304")
    EF1 = st.text_input("EF", "68")

    # 转换用户输入为浮点型
    input_data = pd.DataFrame([[float(PAD1), float(PV1), float(cTnT1), float(DD1), float(Vmax1), float(NEUT1), float(NTproBNP1), float(EF1), float(RAD1)]], columns=['PAD', 'PV', 'cTnT', 'DD', 'Vmax', 'NEUT%', 'NT-proBNP', 'EF', 'RAD'])

    if st.button("Predict"):
        # 获取预测结果
        # result = prediction1(input_data)
        shap_html = draw_force(model_best, input_data)
        result = round(np.exp(shap_html.data["outValue"]) / (1 + np.exp(shap_html.data["outValue"])), 4) * 100
        st.subheader('Based on feature values, predicted possibility of the risk of hospitalization death is: %.2f' % result + "%")
        # 显示 SHAP force plot
        st_shap(shap_html)


if __name__ == '__main__':
    main()

# python -m streamlit run predict.py
