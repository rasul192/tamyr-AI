[![Tamyr AI Verification Pipeline](https://github.com/rasul192/tamyr-AI/actions/workflows/verify.yml/badge.svg)](https://github.com/rasul192/tamyr-AI/actions/workflows/verify.yml)
# Tamyr AI: Physics-Informed Predictive Preservation Engine (v2.0)

Платформа предиктивного анализа и мониторинга объектов культурного наследия Южного Казахстана на основе механики сплошных сред (Continuum Damage Mechanics).

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Запуск проверки ядра
```bash
python run_verification.py
```

### 3. Как использовать в Streamlit (app.py)
```python
import pandas as pd
from src.tamyr_pipeline_integration import TamyrPipeline

df = pd.read_csv("data/turkestan_climate.csv")

pipeline = TamyrPipeline()
result = pipeline.run(df, site_key="yasawi", initial_d=0.28)

print("Статус:", result.risk_status)
print("Итоговый износ D:", result.final_damage)
print("Рекомендации:", result.recommended_methods)
print("Вклад стрессов (%):", result.stress_shares_percent)
print("Фасады:", result.facade_anisotropy)
```

## Структура модулей (src/)
- `tamyr_physics_step.py` — расчет суточного приращения повреждения по 5 стресс-потенциалам.
- `tamyr_integrator.py` — численный интегратор траектории D(t).
- `tamyr_prescriptive_engine.py` — классификация рисков и рекомендации по регламентам ICOMOS.
- `tamyr_facade_microclimate.py` — расчет микроклимата и износа по 4 сторонам света.
- `tamyr_pipeline_integration.py` — мастер-контроллер конвейера для интеграции с дашбордом.
