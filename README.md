# Energy-APP (modular)

Aplicación de pronóstico energético con arquitectura desacoplada:

- `frontend/`: interfaz Streamlit.
- `backend/app/`: API FastAPI para exponer pronósticos.
- `backend/services/`: lógica de negocio (preprocesamiento, modelos y métricas).
- `tests/`: pruebas unitarias de los servicios de pronóstico.

## Estructura

```text
.
├── .github/workflows/ci.yml
├── backend
│   ├── app
│   │   └── main.py
│   └── services
│       ├── evaluation.py
│       ├── forecast.py
│       ├── preprocessing.py
│       ├── rf_model.py
│       ├── sarimax_model.py
│       └── utils.py
├── frontend
│   └── app.py
├── tests
│   └── test_forecast_service.py
├── Makefile
└── app.py
```

## Ejecución

### Frontend (local)

```bash
make run-frontend
```

### API

```bash
make run-api
```

En la interfaz puedes elegir **Modo conexión = API** y apuntar a `http://localhost:8000`.

## Calidad y CI

### Checks locales

```bash
make check
make test
```

### GitHub Actions

El workflow `.github/workflows/ci.yml` ejecuta compilación estática y pruebas unitarias en cada push/PR.

## Despliegue web (considerado)

Se incluyó lógica para desplegar como aplicación web en dos servicios:

- **Backend API** con `backend/Dockerfile` (FastAPI + `uvicorn` en puerto `8000`).
- **Frontend** con `frontend/Dockerfile` (Streamlit en puerto `8501`).
- **Orquestación local** con `docker-compose.yml`.
- **Blueprint de Render** en `deploy/render.yaml`.

### Levantar en local con Docker

```bash
docker compose up --build
```

Luego:
- Frontend: `http://localhost:8501`
- API: `http://localhost:8000/health`

El frontend ahora soporta variables de entorno para despliegue:
- `ENERGY_FORCE_API=1` para iniciar en modo API.
- `ENERGY_API_URL` para definir la URL del backend.

## Guía rápida: desplegar local + actualizar GitHub

### 1) Despliegue local (sin Docker)

```bash
pip install -r requirements.txt
```

Terminal A (API):

```bash
make run-api
```

Terminal B (frontend):

```bash
make run-frontend
```

Abrir en navegador:
- Frontend: `http://localhost:8501`
- Health API: `http://localhost:8000/health`

### 2) Despliegue local con Docker

```bash
docker compose up --build
```

Para detener:

```bash
docker compose down
```

### 3) Actualizar GitHub (subir cambios)

```bash
git status
git add .
git commit -m "tu mensaje de cambio"
git push origin <tu-rama>
```

Si quieres actualizar `main` con PR:
1. `git push origin <tu-rama>`
2. Abrir Pull Request en GitHub
3. Esperar CI en verde
4. Hacer merge a `main`

## Recomendaciones para seguir mejorando en GitHub

1. **Crear Issues por módulos** (`backend`, `frontend`, `data`) con etiquetas (`bug`, `enhancement`, `ml`).
2. **Agregar protección de rama** en `main` (PR obligatorio + CI en verde).
3. **Versionar API** (`/v1/forecast`) para evitar quiebres futuros.
4. **Agregar despliegue** (Render/Railway/Fly.io) para backend y compartir una URL estable con el frontend.
5. **Incorporar tests de integración** para flujo completo `upload -> forecast -> descarga`.
