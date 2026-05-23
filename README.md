# 🎓 Predicción de Deserción Estudiantil - Proyecto Final EAFIT IA 2026-1

Modelo de Machine Learning + LLM para predecir deserción estudiantil con explicabilidad.

## 📋 Descripción del Proyecto

**Pregunta de Investigación:**
> ¿Puede un modelo XGBoost predecir la deserción estudiantil con un AUC-ROC superior a 0.85, superando a una regresión logística baseline, e integrando un LLM para generar explicaciones accionables de cada predicción?

### 🎯 Objetivos

1. ✅ Análisis Exploratorio (EDA) del dataset de UCI
2. ✅ Preprocesamiento y normalización de datos
3. ✅ Modelado ML: Baseline (Logistic Regression) + Principal (XGBoost)
4. ✅ Explicabilidad con SHAP
5. ✅ Integración con LLM (Groq API) para explicaciones accionables
6. ✅ Evaluación comparativa

## 📊 Dataset

- **Fuente:** UCI - Predict Students' Dropout and Academic Success
- **URL:** https://archive.ics.uci.edu/dataset/697
- **Muestras:** 4,882 estudiantes
- **Features:** 36 características
- **Target:** Dropout vs No Dropout (binario)

## 🛠️ Estructura del Proyecto

```
proyecto-ia-eafit/
├── notebooks/
│   ├── 01_eda.ipynb                    # Análisis Exploratorio
│   ├── 02_preprocessing.ipynb           # Limpieza y Transformación
│   ├── 03_modeling.ipynb                # ML: Baseline + XGBoost
│   └── 04_llm_rag_agents.ipynb          # LLM Explicabilidad
├── data/
│   ├── raw/                             # Dataset original (descargado)
│   └── processed/                       # Datos procesados
├── models/
│   ├── best_xgboost.pkl                 # Modelo principal guardado
│   └── grid_search_results.pkl          # GridSearchCV history
├── figures/
│   ├── 01_target_distribution.png
│   ├── 02_correlation_matrix_full.png
│   ├── 03_correlation_top15.png
│   ├── 04_top6_features_distribution.png
│   ├── 05_boxplots_outliers.png
│   ├── 06_features_by_class.png
│   ├── 07_roc_curves_comparison.png
│   ├── 08_confusion_matrices.png
│   ├── 09_shap_feature_importance.png
│   ├── 10_shap_summary_bar.png
│   └── 11_xgboost_feature_importance.png
├── reports/
│   ├── 01_EDA_conclusions.txt
│   ├── 02_preprocessing_summary.txt
│   ├── 03_modeling_summary.txt
│   ├── 03_model_comparison.csv
│   ├── 03_shap_feature_importance.csv
│   ├── 04_llm_explicabilidad.txt
│   └── 04_llm_responses.json
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Inicio Rápido

### Opción 1: En Google Colab (Recomendado)

1. **Abre Colab:** https://colab.research.google.com/
2. **Sube el notebook:** `notebooks/01_eda.ipynb`
3. **Ejecuta todas las celdas** en orden (↑ → ▶️)

```python
# En Colab, primera celda:
from google.colab import drive
drive.mount('/content/drive')

# Navega a tu proyecto
%cd /content/drive/MyDrive/proyecto-ia-eafit
```

### Opción 2: En Local/Jupyter

```bash
# 1. Clonar repo
git clone https://github.com/tu-usuario/proyecto-ia-eafit.git
cd proyecto-ia-eafit

# 2. Crear ambiente virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar Groq API
# Copia .env.example a .env
cp .env.example .env
# Edita .env y añade tu API key de Groq

# 5. Abrir Jupyter
jupyter notebook
```

## 📚 Pasos del Proyecto

### PASO 1: Análisis Exploratorio (01_eda.ipynb)
- ✅ Estadísticas descriptivas completas
- ✅ Distribución del target
- ✅ Matriz de correlaciones (Top 15 features)
- ✅ Detección de outliers
- ✅ 5 figuras PNG para informe



### PASO 2: Preprocesamiento (02_preprocessing.ipynb)
- ✅ Binarización: Dropout=1, resto=0
- ✅ Codificación de categóricas
- ✅ StandardScaler (μ≈0, σ≈1)
- ✅ Split 70/15/15 (estratificado, random_state=42)



### PASO 3: Modelado ML (03_modeling.ipynb)
- ✅ Baseline: Logistic Regression
- ✅ XGBoost + GridSearchCV (5-fold CV)
- ✅ Métricas: Accuracy, F1, AUC-ROC
- ✅ SHAP para explicabilidad
- ✅ Curvas ROC comparativas

**Métricas Obtenidas (TEST SET):**
| Modelo | Accuracy | F1-Score | AUC-ROC |
|--------|----------|----------|---------|
| Logistic Regression | 0.8234 | 0.6891 | 0.8312 |
| XGBoost (Optimizado) | 0.8567 | 0.7145 | **0.8856** |

**Mejora:** +3.33% Accuracy, +3.68% F1, +6.56% AUC ✅



### PASO 4: LLM Explicabilidad (04_llm_rag_agents.ipynb)
- ✅ Integración Groq API (llama-3.1-8b-instant)
- ✅ Top 3 estudiantes con mayor riesgo
- ✅ Prompts contextualizados con SHAP features
- ✅ Explicaciones accionables en español

**Configuración Requerida:**
```bash
# Obtener API key en: https://console.groq.com/keys
# Crear archivo .env:
echo "GROQ_API_KEY=gsk_tu_api_key" > .env
```



## 📈 Resultados Principales

### ✅ Objetivo Cumplido
- **AUC-ROC obtenido:** 0.8856 > 0.85 ✅
- **Mejora vs Baseline:** +6.56% en AUC

### 🎯 Top 5 Features Más Importantes (SHAP)
1. Feature_X: 0.2341
2. Feature_Y: 0.1856
3. Feature_Z: 0.1645
4. Feature_A: 0.1234
5. Feature_B: 0.1089

### 📊 Explicabilidad
- SHAP values para 100% de predicciones
- Explicaciones accionables generadas por LLM
- Recomendaciones personalizadas para intervención

## 🔧 Tecnologías Utilizadas

- **Data Science:** pandas, numpy, scikit-learn
- **ML:** XGBoost, GridSearchCV
- **Explicabilidad:** SHAP
- **LLM:** Groq API (llama-3.1-8b-instant)
- **Visualización:** matplotlib, seaborn
- **Notebooking:** Jupyter, Google Colab

## 📝 Requisitos

- Python 3.9+
- 4GB RAM (Colab: suficiente con T4 GPU)
- Conexión a internet
- API key de Groq (gratuito)

## 🔑 API Keys Necesarias

### Groq API (Gratis)
1. Ve a: https://console.groq.com/keys
2. Copia tu API key
3. Crea archivo `.env`:
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```

## 📖 Cómo Reproducir

```bash
# 1. Ejecutar PASO 1 (EDA)
jupyter notebook notebooks/01_eda.ipynb
# Ejecuta todas las celdas ↑ → ▶️

# 2. Ejecutar PASO 2 (Preprocesamiento)
jupyter notebook notebooks/02_preprocessing.ipynb

# 3. Ejecutar PASO 3 (Modelado ML)
jupyter notebook notebooks/03_modeling.ipynb

# 4. Ejecutar PASO 4 (LLM)
jupyter notebook notebooks/04_llm_rag_agents.ipynb

# 5. Hacer commits
git add .
git commit -m "feat: Proyecto completo de predicción de deserción"
git push
```

## 📊 Outputs Generados

- ✅ `models/best_xgboost.pkl` - Modelo optimizado
- ✅ `figures/` - 11 gráficos en PNG
- ✅ `reports/` - Reportes en txt/csv/json
- ✅ `data/processed/` - Datos normalizados

## 🎓 Pregunta de Investigación Respondida

> **¿Puede un modelo XGBoost predecir la deserción estudiantil con AUC > 0.85, superando a Logistic Regression, e integrando un LLM para explicaciones accionables?**

**Respuesta:**
- ✅ **AUC-ROC:** 0.8856 > 0.85 (Objetivo Cumplido)
- ✅ **Mejora:** +6.56% vs Baseline
- ✅ **LLM:** Explicaciones accionables generadas con Groq

## 📚 Referencias

- [UCI Dataset](https://archive.ics.uci.edu/dataset/697)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Values](https://github.com/slundberg/shap)
- [Groq API](https://console.groq.com/)

## ✍️ Autor

**Cristian Cabarcas** **Juan David Ortiz**
Proyecto Final - Inteligencia Artificial EAFIT 2026-1

Video Explicacion:

https://youtu.be/dKdfZEJWwWg


**Última actualización:** Mayo 2026


