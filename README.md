# Señas — reconocimiento de dactilología en tiempo real

Aplicación web para practicar el alfabeto dactilológico de la **Lengua de Señas Mexicana (LSM)** o del **American Sign Language (ASL)**. Usa la cámara del navegador, detecta 21 puntos de la mano con MediaPipe y reconoce **todas las letras del alfabeto desde el primer uso** con una base geométrica; tus muestras entrenan un ajuste personal que mejora la precisión. También forma palabras que puedes escuchar, guardar y repasar.

Todo corre en el navegador. No hay backend, no se sube video y no se graban fotos ni audio.

---

## 1. Cómo ejecutarla

La cámara requiere un **contexto seguro**: `https://` o `localhost`. Abrir `index.html` con doble clic (`file://`) no permite usar la cámara.

### Servidor local

Desde la carpeta del proyecto:

```bash
python -m http.server 8000
```

Abre <http://localhost:8000>.

Alternativas:

```bash
npx serve .
# o
php -S localhost:8000
```

### GitHub Pages

1. Sube `index.html`, `estilos.css`, `scrpts.js` y este `README.md` a un repositorio.
2. En GitHub: **Settings → Pages → Build and deployment**.
3. Selecciona **Deploy from a branch**, rama `main` y carpeta `/ (root)`.
4. Espera a que GitHub publique el sitio y ábrelo por HTTPS.

La app usa rutas relativas, así que funciona igual en la raíz de un dominio o en un subdirectorio (`usuario.github.io/repositorio/`).

> El nombre `scrpts.js` se conserva tal como estaba en el proyecto original.

---

## 2. Primeros pasos

1. Pulsa **Activar cámara** y acepta el permiso.
2. Ve a **Reconocer** y muestra una letra. La base general intenta identificarla aunque no hayas entrenado nada.
3. Si una letra se confunde, ve a **Entrenar**, elige esa letra y mantén pulsado **Capturar muestra** mientras haces la seña. La app toma varias muestras por segundo.
4. Repite con distintos ángulos, distancias e iluminación hasta llegar a **30–50 muestras** por letra (el máximo es 200). Esas muestras afinan el reconocimiento de tu estilo.
5. Vuelve a **Reconocer**, muestra la seña y espera a que la barra de estabilización se complete.

Cada letra se marca como **sin personalizar** (0 muestras, usa solo la base general), **en progreso** (1–29) o **personalizada** (30 o más). Entrenar ya no “habilita” letras: solo mejora la precisión de las que practicas.

### Modo práctica

En **Reconocer**, pulsa **Iniciar práctica**. La app propone una letra entrenada al azar y suma puntos si la reconoces. La práctica **no añade letras a la palabra actual**. Puedes saltar de letra con **Otra letra**.

### Diccionario e historial

- **Guardar palabra** añade la palabra actual al diccionario local y al historial.
- Las palabras del diccionario alimentan el **autocompletado** mientras deletreas.
- Cada palabra guardada tiene acciones para pronunciarla, reutilizarla o eliminarla.
- **Historial** conserva las últimas 40 frases guardadas.
- La pronunciación usa `speechSynthesis` con una voz en español si está instalada en el sistema.

---

## 3. Cómo funciona

### Detección

- `@tensorflow-models/hand-pose-detection` con el modelo **MediaPipeHands**.
- Se intenta primero el runtime **MediaPipe**; si falla, se usa el runtime **tfjs**.
- Se procesa **una mano** y se dibujan las 21 articulaciones y 21 conexiones sobre un canvas espejado.
- El color del esqueleto cambia según la confianza de la predicción.

### Normalización

Cada mano se convierte en un vector de **42 números** (21 puntos × 2 coordenadas):

1. Se toma la muñeca como origen.
2. Se divide por el tamaño de la palma (distancias muñeca–nudillos).
3. Si la mano detectada es izquierda, se invierte el eje X para compensar el espejo.
4. El resultado es independiente de la posición y el tamaño de la mano en la imagen.

### Clasificación

La predicción combina dos capas:

1. **Base general (sin entrenamiento).** Cada mano se describe con 21 características geométricas: rectitud y extensión de los cuatro dedos, separación entre dedos, posición y alcance del pulgar, orientación de la palma y movimiento. Esas características se comparan con plantillas de las 27 letras (A–Z y Ñ en LSM) y se obtiene un puntaje para cada una. Esta capa funciona desde el primer uso y no depende de muestras.
2. **Ajuste personal (opcional).** Si entrenaste una letra, sus muestras se comparan con la seña actual. Solo cuentan si están razonablemente cerca; una letra entrenada no puede “robar” una seña que está lejos de sus muestras. El peso del ajuste personal crece con el número de muestras (hasta un máximo), de modo que el entrenamiento afina la precisión sin desactivar las demás letras.

El valor **K** (1–15) controla cuántos vecinos personales se promedian por letra. La confianza combina la calidad del mejor puntaje con la ventaja sobre la segunda letra.

Si ninguna plantilla encaja o dos letras quedan demasiado empatadas, la app muestra **«Seña no reconocida»** y no agrega ninguna letra. Así evita clasificar automáticamente una seña desconocida como otra que sí estaba personalizada.

### Estabilización

Para evitar letras accidentales:

- La misma letra debe aparecer durante **N frames consecutivos** (5–40; por defecto 15).
- Debe superar el **umbral de confianza** configurable (50–99 %; por defecto 70 %).
- Si la seña no se reconoce o está empatada, se muestra **«Seña no reconocida»** y el contador se reinicia.
- Después de aceptar una letra, hay que **cambiar de seña o retirar la mano** para aceptar la siguiente.
- Se muestra una barra de progreso y el contador de frames.

### Letras con movimiento: J, Z y Ñ

J, Z y la Ñ de LSM requieren un trazo en el aire. La app:

- Guarda la trayectoria de la punta del índice.
- Calcula la amplitud del movimiento respecto al tamaño de la palma.
- Solo acepta estas letras si la amplitud supera `0.24` y la opción **Exigir movimiento** está activa.
- Dibuja un rastro rosa de la punta del índice.

Es una aproximación **experimental**: no reconoce la forma exacta de la J/Z, solo que hubo movimiento suficiente. Para uso serio conviene entrenar estas letras con trayectorias variadas o desactivar la exigencia de movimiento y tratarlas como letras estáticas.

---

## 4. Privacidad y datos

- **No se graba video, audio ni fotos.** La cámara se usa solo para extraer los 21 puntos de la mano en memoria.
- **No hay servidor ni subida de datos.** Todo el procesamiento ocurre en el navegador.
- Se guardan en `localStorage` del navegador:
  - `senas.dataset.v1`: muestras por alfabeto y letra.
  - `senas.diccionario.v1`: palabras y número de usos.
  - `senas.historial.v1`: últimas frases.
  - `senas.config.v1`: alfabeto, K, umbral, frames, cámara y preferencia de movimiento.
- Al cargar la página se descargan las librerías desde jsDelivr y las fuentes desde Google Fonts. Eso sí requiere internet la primera vez (y respeta la caché del navegador).
- El botón **Exportar** genera un JSON con el dataset. **Importar** lo combina con el actual, sin borrar lo existente.

### Formato del JSON exportado

```json
{
  "app": "senas",
  "version": 1,
  "exportado": "2026-11-01T18:00:00.000Z",
  "alfabeto": "lsm",
  "datos": {
    "lsm": { "A": [[0.1, 0.2, 0.3]] },
    "asl": {}
  }
}
```

Cada muestra del arreglo debe contener exactamente **42 números finitos**; el ejemplo se abrevia. La importación acepta también un objeto que solo contenga `datos` o directamente los alfabetos. Se descartan letras desconocidas, vectores que no tengan exactamente 42 números finitos y muestras por encima de 200 por letra.

---

## 5. Limitaciones conocidas

- **No interpreta lengua de señas completa.** Reconoce letras del alfabeto dactilológico; no traduce gramática, expresiones faciales ni señas con significado propio.
- **La base general es aproximada.** Las letras se reconocen desde el primer uso, pero la geometría de MediaPipe no distingue perfectamente todas las configuraciones (por ejemplo M/N/E, U/V o G/Q). Entrenar esa letra con 30–50 muestras es lo que mejora su precisión.
- **Una sola mano**, cámara frontal. No se usan dos manos ni el cuerpo.
- **Letras estáticas.** J, Z y Ñ dependen de un umbral de movimiento; la forma del trazo no se valida.
- **Depende de la persona y la cámara.** La base general y las muestras pueden fallar con otra persona, otra cámara o mucha diferencia de luz. Puedes exportar el dataset e importarlo, pero no es un modelo universal.
- **Señas parecidas** (M/N, U/V, R, etc.) pueden confundirse. Más muestras y ángulos variados ayudan; si la app no está segura, avisa en lugar de adivinar.
- **Fondo y luz:** una mano completa, bien iluminada y con fondo simple funciona mejor.
- **Rendimiento:** el detector corre en el navegador. En equipos modestos o sin WebGL puede ir más lento; el runtime tfjs es el respaldo.
- **iOS/Safari** puede limitar cámaras o requerir HTTPS estricto. Chrome, Edge y Firefox actualizados son la referencia.
- **Voz:** si no hay voces en español instaladas, el navegador usa la voz por defecto o no pronuncia.
- **No hay sincronización en la nube.** Si borras los datos del navegador, se pierde el entrenamiento; exporta un respaldo.

---

## 6. Solución de problemas

| Síntoma | Qué revisar |
| --- | --- |
| «La cámara requiere HTTPS o localhost» | Abre la app con un servidor local o HTTPS, no con `file://`. |
| No aparece el permiso de cámara | Revisa el icono de candado del navegador y los permisos del sistema. |
| «Modelo no disponible» | Comprueba internet y que `cdn.jsdelivr.net` no esté bloqueado. Pulsa **Reintentar**. |
| El modelo carga pero no detecta la mano | Acércala, muestra la palma completa, mejora la luz y evita fondos con manos. |
| Reconoce letras equivocadas | La base general es aproximada. Entrena esa letra con 30–50 muestras, varía ángulos y distancia, sube el umbral o ajusta K. |
| Aparece «Seña no reconocida» | La pose no encaja con ninguna plantilla o hay empate entre letras. Corrige la forma, acerca la mano o personaliza esa letra con muestras. |
| Se aceptan letras sin querer | Sube **Frames estables**, sube la confianza mínima o revisa que la mano esté quieta. |
| J/Z/Ñ no se aceptan | Activa **Exigir movimiento** solo si harás el trazo; si no, desactívalo. |
| No hay sonido | Instala una voz en español en el sistema y revisa el volumen. |
| Se perdió el entrenamiento | Los datos viven en este navegador; importa un JSON exportado o vuelve a entrenar. |

---

## 7. Estructura del proyecto

```text
señas/
├─ index.html    Interfaz, pestañas, paneles, diálogos y carga de CDNs
├─ estilos.css   Tema oscuro, diseño responsive, animaciones y accesibilidad
├─ scrpts.js     Cámara, MediaPipe, normalización, clasificación híbrida, palabra, diccionario y práctica
└─ README.md     Esta guía
```

### Funciones principales de `scrpts.js`

- `cargarModelo()` / `crearDetector()`: carga MediaPipe Hands con respaldo tfjs.
- `activarCamara()` / `detenerCamara()`: ciclo de vida de `getUserMedia`.
- `normalizarLandmarks()`: vector de 42 valores.
- `extraerCaracteristicas()` / `puntuarCaracteristicas()`: base general geométrica para todas las letras.
- `predecirPersonalizado()`: compara con tus muestras y calcula similitudes por letra.
- `predecir()`: mezcla base general y ajuste personal; devuelve la letra o «no reconocida».
- `reconstruirClasificador()`: reconstruye el índice de muestras personales desde `localStorage`.
- `procesarManos()`: pipeline por frame (dibujo, movimiento, predicción, estabilización).
- `capturarMuestra()`: añade una muestra a la letra seleccionada.
- `evaluarEstabilizacion()` / `aceptarLetra()`: control de aceptación.
- `agregarLetraPalabra()` / `guardarPalabraActual()`: formación y guardado.
- `iniciarPractica()` / `evaluarPractica()`: modo práctica.
- `importarDataset()` / `exportarDataset()`: respaldo JSON.

---

## 8. Lista corta de pruebas manuales

1. Abrir `http://localhost:8000` y ver «Modelo listo».
2. Activar la cámara y comprobar el video espejado y el esqueleto de 21 puntos.
3. Sin entrenar nada, mostrar una letra (por ejemplo A o B) y comprobar que la base general la detecta.
4. Entrenar la letra A con 30 muestras; ver el contador y el indicador «personalizada».
5. Reconocer A manteniendo la seña; comprobar barra, letra y confianza, y que el origen diga «Personalizada».
6. Mostrar una letra sin entrenar (por ejemplo E) y comprobar que no se clasifica como A o B.
7. Mostrar una pose ambigua o fuera del alfabeto y comprobar el aviso «Seña no reconocida».
8. Cambiar de seña y comprobar que no se repite la misma letra sin retirar la mano.
9. Formar «HOLA MUNDO», guardarla y verla en Diccionario e Historial.
10. Escribir «HO» y comprobar la sugerencia «HOLA MUNDO».
11. Iniciar práctica, acertar una letra y comprobar puntos y racha.
12. Exportar el dataset, borrar una letra y volver a importar el JSON.
13. Recargar la página y comprobar que dataset, diccionario, historial y ajustes siguen ahí.
14. Probar J/Z/Ñ con y sin movimiento.
15. Reducir la ventana a 390 px y comprobar que las pestañas y paneles no se cortan.
16. Detener y reactivar la cámara; comprobar que el selector de cámara se comporta bien.

---

## 9. Estado y trabajo pendiente

**Implementado:** detección en tiempo real, normalización, reconocimiento general de todo el alfabeto sin entrenamiento, ajuste personal por muestras, rechazo de señas no reconocidas, entrenamiento con captura sostenida, estabilización, palabra, diccionario, autocompletado, historial, voz, práctica, LSM/ASL, exportación/importación, persistencia, accesibilidad, diseño responsive y manejo de errores de cámara/modelo.

**Pendiente crítico:** ninguno para el alcance actual.

**Mejoras opcionales para el futuro:**

- Seguimiento de dos manos y señas dinámicas reales (no solo umbral de movimiento).
- Modelo preentrenado con datos de muchas personas para una base general más precisa.
- PWA con caché offline de las librerías.
- Editor de dataset con vista previa y limpieza selectiva de muestras.
- Métricas de precisión por letra y recomendaciones automáticas de muestras.
- Pruebas automatizadas dentro del repositorio (las actuales se ejecutaron con Chrome/CDP).

---

## 10. Créditos y versiones

- [TensorFlow.js](https://www.tensorflow.org/js) `4.22.0`
- [Hand Pose Detection](https://github.com/tensorflow/tfjs-models/tree/master/hand-pose-detection) `2.0.1`
- [MediaPipe Hands](https://github.com/google/mediapipe) `0.4.1675469240`
- Fuentes: Sora y JetBrains Mono (Google Fonts)

Herramienta educativa. No sustituye a un intérprete profesional ni a una persona instructora de LSM/ASL. Las configuraciones de las señas varían entre escuelas y regiones; valida el alfabeto con una persona sorda o docente de tu comunidad.
