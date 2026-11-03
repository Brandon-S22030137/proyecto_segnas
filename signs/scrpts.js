/* ============================================================
   Señas · Reconocimiento de lenguaje de señas en el navegador
   ------------------------------------------------------------
   Arquitectura (todo en un archivo, con funciones separadas):
   1. Datos del alfabeto y constantes
   2. Estado de la aplicación
   3. Almacenamiento local (dataset, diccionario, historial, config)
   4. Cámara y modelo de detección de manos
   5. Normalización de landmarks y clasificación híbrida
   6. Estabilización y formación de palabras
   7. Entrenamiento y gestión del dataset
   8. Diccionario, historial, voz y modo práctica
   9. Interfaz, accesibilidad y arranque

   Privacidad: el video nunca se graba ni se envía. Solo se guardan
   vectores normalizados de 42 números (21 puntos × x,y).
   ============================================================ */

'use strict';

/* ============================================================
   1. DATOS DEL ALFABETO Y CONSTANTES
   ============================================================ */

/**
 * Descripciones orientativas del alfabeto dactilológico.
 * Las configuraciones cambian según la escuela y la región; la app
 * aprende de las muestras del usuario, no de estas descripciones.
 */
const SENAS_BASE = Object.freeze({
  A: 'Puño cerrado con el pulgar pegado al costado del índice.',
  B: 'Mano abierta, dedos juntos y estirados; pulgar cruzado sobre la palma.',
  C: 'Mano curvada formando una C, con la palma hacia un lado.',
  D: 'Índice hacia arriba; los demás dedos tocan el pulgar formando un círculo.',
  E: 'Dedos doblados hacia la palma y pulgar por debajo de ellos.',
  F: 'Pulgar e índice forman un círculo; los otros tres dedos extendidos.',
  G: 'Índice y pulgar extendidos y paralelos, apuntando de lado.',
  H: 'Índice y medio extendidos y juntos, apuntando de lado.',
  I: 'Meñique extendido hacia arriba; el resto en puño.',
  J: 'Meñique extendido; traza una J en el aire con la punta.',
  K: 'Índice y medio arriba en V; pulgar entre ambos.',
  L: 'Índice arriba y pulgar extendido al lado, formando una L.',
  M: 'Pulgar bajo tres dedos doblados: índice, medio y anular.',
  N: 'Pulgar bajo dos dedos doblados: índice y medio.',
  O: 'Todos los dedos curvados hasta tocar el pulgar, formando una O.',
  P: 'Como la K pero con la mano apuntando hacia abajo.',
  Q: 'Como la G pero con la mano apuntando hacia abajo.',
  R: 'Índice y medio extendidos y cruzados.',
  S: 'Puño cerrado con el pulgar por delante de los dedos.',
  T: 'Pulgar entre el índice y el medio, dentro del puño.',
  U: 'Índice y medio extendidos y juntos, apuntando arriba.',
  V: 'Índice y medio extendidos y separados en V.',
  W: 'Índice, medio y anular extendidos y separados.',
  X: 'Índice doblado en gancho; los demás dedos en puño.',
  Y: 'Pulgar y meñique extendidos; índice, medio y anular cerrados.',
  Z: 'Índice extendido; traza una Z en el aire.'
});

const LETRAS_ASL = Object.freeze(Object.keys(SENAS_BASE));
const LETRAS_LSM = Object.freeze([
  ...LETRAS_ASL.slice(0, 14),
  'Ñ',
  ...LETRAS_ASL.slice(14)
]);

const SENAS_LSM = Object.freeze({
  ...SENAS_BASE,
  Ñ: 'Como la N, añadiendo un pequeño movimiento ondulado de la muñeca (tilde).'
});

const ALFABETOS = Object.freeze({
  lsm: Object.freeze({
    id: 'lsm',
    nombre: 'LSM',
    descripcion: 'Lengua de Señas Mexicana · A–Z + Ñ',
    letras: LETRAS_LSM,
    senas: SENAS_LSM
  }),
  asl: Object.freeze({
    id: 'asl',
    nombre: 'ASL',
    descripcion: 'American Sign Language · A–Z',
    letras: LETRAS_ASL,
    senas: SENAS_BASE
  })
});

/* Letras que dependen de una trayectoria, no solo de una pose estática. */
const LETRAS_MOVIMIENTO = new Set(['J', 'Z', 'Ñ']);

/* Reconocimiento general (sin entrenamiento) y mezcla con las muestras del usuario. */
const UMBRAL_FORMA_MINIMA = 0.34;
const MARGEN_MINIMO = 0.035;
const RADIO_SIMILITUD_PERSONAL = 0.7;
const MAX_PESO_PERSONAL = 0.92;
const PESO_PERSONAL_INICIAL = 0.25;
const PESO_PERSONAL_COMPLETO = 0.67;

const CLAVES = Object.freeze({
  dataset: 'senas.dataset.v1',
  diccionario: 'senas.diccionario.v1',
  historial: 'senas.historial.v1',
  config: 'senas.config.v1'
});

const CONFIG_INICIAL = Object.freeze({
  alfabeto: 'lsm',
  k: 5,
  umbral: 0.7,
  frames: 15,
  exigirMovimiento: true,
  camaraId: ''
});

const MIN_MUESTRAS = 30;
const MAX_MUESTRAS = 200;
const VENTANA_MOVIMIENTO = 24;
const UMBRAL_MOVIMIENTO = 0.24;
const MAX_PALABRA = 40;
const MAX_HISTORIAL = 40;

/* Conexiones del esqueleto de la mano (21 landmarks de MediaPipe). */
const CONEXIONES = Object.freeze([
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
]);

const PUNTAS = new Set([4, 8, 12, 16, 20]);

/* ============================================================
   2. ESTADO DE LA APLICACIÓN
   ============================================================ */

const estado = {
  modeloListo: false,
  cargandoModelo: false,
  runtimeModelo: '',
  detector: null,

  camaraActiva: false,
  stream: null,
  rafId: null,
  avisoDeteccion: false,

  tabActiva: 'reconocer',
  dataset: { lsm: {}, asl: {} },
  config: { ...CONFIG_INICIAL },
  diccionario: {},
  historial: [],
  almacenamientoOk: true,
  avisoCuota: false,

  /* Muestras personales aplanadas para el clasificador (letra + vector de 42). */
  muestrasClasificador: [],
  conteoLetras: new Map(),

  /* Frame actual */
  vector: null,
  mano: false,
  manoUI: null,
  handedness: '',
  confianza: 0,
  prediccion: null,
  bufferPunta: [],
  movimiento: 0,
  framesSinMano: 0,

  /* Estabilización */
  candidato: '',
  contador: 0,
  ultimaLetra: '',
  esperandoCambio: false,

  /* Entrenamiento */
  letraEntrenamiento: null,
  capturando: false,
  intervaloCaptura: null,
  ultimaCaptura: 0,

  /* Palabra y práctica */
  palabra: '',
  practica: {
    activa: false,
    objetivo: '',
    puntos: 0,
    racha: 0,
    mejor: 0
  },

  temporizadorGuardado: null
};

let dom = {};

/* ============================================================
   3. ALMACENAMIENTO LOCAL
   ============================================================ */

function cachearDOM() {
  const $ = (id) => document.getElementById(id);

  dom = {
    chipModelo: $('chip-modelo'),
    estadoModelo: $('estado-modelo'),
    selectAlfabeto: $('select-alfabeto'),
    tabs: Array.from(document.querySelectorAll('.tab')),
    vistas: Array.from(document.querySelectorAll('.vista')),

    video: $('video'),
    canvas: $('canvas'),
    capaCamara: $('capa-camara'),
    capaIcono: $('capa-icono'),
    capaTitulo: $('capa-titulo'),
    capaTexto: $('capa-texto'),
    capaBoton: $('capa-boton'),
    avisoSinMano: $('aviso-sin-mano'),
    estadoCamara: $('estado-camara'),
    textoEstadoCamara: $('texto-estado-camara'),
    selectCamara: $('select-camara'),
    btnDetenerCamara: $('btn-detener-camara'),

    letraDetectada: $('letra-detectada'),
    confianzaValor: $('confianza-valor'),
    barraConfianza: $('barra-confianza'),
    estadoMano: $('estado-mano'),
    textoEstadoMano: $('texto-estado-mano'),
    estadoLateralidad: $('estado-lateralidad'),
    origenPrediccion: $('origen-prediccion'),
    textoEstabilizacion: $('texto-estabilizacion'),
    contadorEstabilizacion: $('contador-estabilizacion'),
    progresoEstabilizacion: $('progreso-estabilizacion'),
    barraEstabilizacion: $('barra-estabilizacion'),
    valorMovimiento: $('valor-movimiento'),
    barraMovimiento: $('barra-movimiento'),

    rangoK: $('rango-k'),
    valorK: $('valor-k'),
    rangoUmbral: $('rango-umbral'),
    valorUmbral: $('valor-umbral'),
    rangoFrames: $('rango-frames'),
    valorFrames: $('valor-frames'),
    checkMovimiento: $('check-movimiento'),

    palabraActual: $('palabra-actual'),
    sugerencias: $('sugerencias'),
    btnEspacio: $('btn-espacio'),
    btnBorrarLetra: $('btn-borrar-letra'),
    btnLimpiarPalabra: $('btn-limpiar-palabra'),
    btnGuardarPalabra: $('btn-guardar-palabra'),
    btnPronunciarPalabra: $('btn-pronunciar-palabra'),

    practicaPanel: $('practica-panel'),
    practicaEstado: $('practica-estado'),
    practicaObjetivo: $('practica-objetivo'),
    practicaPuntos: $('practica-puntos'),
    practicaRacha: $('practica-racha'),
    practicaMejor: $('practica-mejor'),
    practicaFeedback: $('practica-feedback'),
    btnPractica: $('btn-practica'),
    btnPracticaSaltar: $('btn-practica-saltar'),

    alfabetoEntrenar: $('alfabeto-entrenar'),
    videoEntrenar: $('video-entrenar'),
    canvasEntrenar: $('canvas-entrenar'),
    capaEntrenar: $('capa-entrenar'),
    textoCapaEntrenar: $('texto-capa-entrenar'),
    btnCamaraEntrenar: $('btn-camara-entrenar'),
    letraSeleccionada: $('letra-seleccionada'),
    contadorLetra: $('contador-letra'),
    metaLetra: $('meta-letra'),
    barraLetra: $('barra-letra'),
    btnCapturar: $('btn-capturar'),
    gridLetras: $('grid-letras'),
    numConMuestras: $('num-con-muestras'),
    numListas: $('num-listas'),
    numPendientes: $('num-pendientes'),
    btnExportar: $('btn-exportar'),
    btnImportar: $('btn-importar'),
    inputImportar: $('input-importar'),
    btnBorrarLetra: $('btn-borrar-letra'),
    btnBorrarTodo: $('btn-borrar-todo'),

    contadorDiccionario: $('contador-diccionario'),
    formDiccionario: $('form-diccionario'),
    inputPalabra: $('input-palabra'),
    listaDiccionario: $('lista-diccionario'),
    vacioDiccionario: $('vacio-diccionario'),
    listaHistorial: $('lista-historial'),
    vacioHistorial: $('vacio-historial'),
    btnLimpiarHistorial: $('btn-limpiar-historial'),

    nombreAlfabetoAyuda: $('nombre-alfabeto-ayuda'),
    referenciaAlfabeto: $('referencia-alfabeto'),

    toasts: $('toasts'),
    anuncio: $('anuncio'),
    dialogoConfirmar: $('dialogo-confirmar'),
    dialogoTitulo: $('dialogo-titulo'),
    dialogoMensaje: $('dialogo-mensaje'),
    dialogoBtnConfirmar: $('dialogo-btn-confirmar')
  };
}

function probarAlmacenamiento() {
  try {
    const clave = '__senas_prueba__';
    localStorage.setItem(clave, '1');
    localStorage.removeItem(clave);
    return true;
  } catch (error) {
    console.warn('localStorage no disponible:', error);
    return false;
  }
}

function leerJSON(clave, respaldo) {
  try {
    const crudo = localStorage.getItem(clave);
    return crudo ? JSON.parse(crudo) : respaldo;
  } catch (error) {
    console.warn('No se pudo leer', clave, error);
    return respaldo;
  }
}

function guardarJSON(clave, valor) {
  if (!estado.almacenamientoOk) return false;

  try {
    localStorage.setItem(clave, JSON.stringify(valor));
    return true;
  } catch (error) {
    console.warn('No se pudo guardar', clave, error);
    if (!estado.avisoCuota) {
      estado.avisoCuota = true;
      mostrarToast('No se pudo guardar: almacenamiento lleno o bloqueado. Exporta el dataset.', 'error', 6500);
    }
    return false;
  }
}

function esVector(valor) {
  return Array.isArray(valor) &&
    valor.length === 42 &&
    valor.every((numero) => Number.isFinite(numero));
}

function sanearAlfabeto(crudo, alfabeto) {
  const salida = {};
  if (!crudo || typeof crudo !== 'object') return salida;

  for (const letra of ALFABETOS[alfabeto].letras) {
    const muestras = crudo[letra];
    if (!Array.isArray(muestras)) continue;
    const validas = muestras.filter(esVector).slice(0, MAX_MUESTRAS);
    if (validas.length) salida[letra] = validas;
  }

  return salida;
}

function cargarDatos() {
  const datasetCrudo = leerJSON(CLAVES.dataset, null);
  if (datasetCrudo && typeof datasetCrudo === 'object') {
    const fuente = datasetCrudo.datos && typeof datasetCrudo.datos === 'object'
      ? datasetCrudo.datos
      : datasetCrudo;
    estado.dataset.lsm = sanearAlfabeto(fuente.lsm, 'lsm');
    estado.dataset.asl = sanearAlfabeto(fuente.asl, 'asl');
  }

  const configCruda = leerJSON(CLAVES.config, null);
  if (configCruda && typeof configCruda === 'object') {
    if (ALFABETOS[configCruda.alfabeto]) estado.config.alfabeto = configCruda.alfabeto;
    estado.config.k = limitar(Number(configCruda.k) || CONFIG_INICIAL.k, 1, 15);
    estado.config.umbral = limitar(Number(configCruda.umbral) || CONFIG_INICIAL.umbral, 0.5, 0.99);
    estado.config.frames = limitar(Number(configCruda.frames) || CONFIG_INICIAL.frames, 5, 40);
    estado.config.exigirMovimiento = configCruda.exigirMovimiento !== false;
    estado.config.camaraId = typeof configCruda.camaraId === 'string' ? configCruda.camaraId : '';
  }

  const diccionarioCrudo = leerJSON(CLAVES.diccionario, {});
  const diccionarioLimpio = {};
  if (diccionarioCrudo && typeof diccionarioCrudo === 'object') {
    for (const [palabra, datos] of Object.entries(diccionarioCrudo)) {
      const normalizada = normalizarPalabra(palabra);
      if (!normalizada) continue;
      const usos = Number(datos && datos.usos);
      diccionarioLimpio[normalizada] = {
        usos: Number.isFinite(usos) ? Math.max(0, Math.floor(usos)) : 0,
        ultima: Number(datos && datos.ultima) || Date.now()
      };
    }
  }
  estado.diccionario = diccionarioLimpio;

  const historialCrudo = leerJSON(CLAVES.historial, []);
  if (Array.isArray(historialCrudo)) {
    estado.historial = historialCrudo
      .map((entrada) => {
        if (typeof entrada === 'string') {
          return { texto: normalizarPalabra(entrada), fecha: Date.now() };
        }
        return {
          texto: normalizarPalabra(entrada && entrada.texto),
          fecha: Number(entrada && entrada.fecha) || Date.now()
        };
      })
      .filter((entrada) => entrada.texto)
      .slice(0, MAX_HISTORIAL);
  }
}

function guardarConfig() {
  guardarJSON(CLAVES.config, estado.config);
}

function guardarDiccionario() {
  guardarJSON(CLAVES.diccionario, estado.diccionario);
}

function guardarHistorial() {
  guardarJSON(CLAVES.historial, estado.historial);
}

function guardarDataset() {
  if (estado.temporizadorGuardado) {
    clearTimeout(estado.temporizadorGuardado);
  }

  estado.temporizadorGuardado = window.setTimeout(() => {
    estado.temporizadorGuardado = null;
    guardarJSON(CLAVES.dataset, estado.dataset);
  }, 320);
}

function volcarDatasetInmediato() {
  if (estado.temporizadorGuardado) {
    clearTimeout(estado.temporizadorGuardado);
    estado.temporizadorGuardado = null;
  }
  try {
    localStorage.setItem(CLAVES.dataset, JSON.stringify(estado.dataset));
  } catch (error) {
    console.warn('No se pudo volcar el dataset:', error);
  }
}

/* ============================================================
   4. CÁMARA Y MODELO
   ============================================================ */

async function cargarModelo() {
  if (estado.cargandoModelo || estado.modeloListo) return;

  estado.cargandoModelo = true;
  actualizarChipModelo('cargando', 'Cargando modelo…');
  dom.btnCamaraEntrenar.disabled = true;
  dom.textoCapaEntrenar.textContent = 'Cargando el modelo de manos…';
  dom.btnCamaraEntrenar.textContent = 'Espera un momento…';

  mostrarCapa({
    tipo: 'info',
    icono: '⏳',
    titulo: 'Cargando modelo de manos',
    texto: 'Preparando la detección de 21 puntos. La primera vez descarga TensorFlow.js y MediaPipe desde el CDN.',
    boton: 'Espera un momento…',
    deshabilitado: true
  });

  try {
    if (!window.tf || !window.handPoseDetection) {
      throw new Error('Las librerías de TensorFlow.js no están disponibles.');
    }

    estado.detector = await crearDetector();
    estado.modeloListo = true;
    actualizarChipModelo('listo', 'Modelo listo');
    dom.chipModelo.title = `Detección de manos: ${estado.runtimeModelo}`;
    dom.btnCamaraEntrenar.disabled = false;
    dom.textoCapaEntrenar.textContent = 'Activa la cámara para ver el esqueleto mientras capturas.';
    dom.btnCamaraEntrenar.textContent = 'Activar cámara';
    mostrarCapaInicio();
    anunciar('Modelo listo. Activa la cámara para empezar.');
  } catch (error) {
    console.error('No se pudo cargar el modelo:', error);
    estado.modeloListo = false;
    actualizarChipModelo('error', 'Modelo no disponible');
    dom.btnCamaraEntrenar.disabled = true;
    dom.textoCapaEntrenar.textContent = 'No se pudo cargar el modelo. Revisa tu conexión y recarga la página.';
    dom.btnCamaraEntrenar.textContent = 'Modelo no disponible';

    mostrarCapa({
      tipo: 'error',
      icono: '⚠️',
      titulo: 'No se pudo cargar el modelo',
      texto: 'Revisa tu conexión a internet y que el CDN de jsDelivr no esté bloqueado. La primera carga descarga TensorFlow.js y MediaPipe.',
      boton: 'Reintentar',
      accion: cargarModelo
    });
  } finally {
    estado.cargandoModelo = false;
  }
}

async function crearDetector() {
  const modelo = handPoseDetection.SupportedModels.MediaPipeHands;

  try {
    const detector = await handPoseDetection.createDetector(modelo, {
      runtime: 'mediapipe',
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240',
      modelType: 'full',
      maxHands: 1
    });
    estado.runtimeModelo = 'mediapipe';
    return detector;
  } catch (error) {
    console.warn('El runtime MediaPipe no está disponible; probando el runtime tfjs.', error);
    const detector = await handPoseDetection.createDetector(modelo, {
      runtime: 'tfjs',
      modelType: 'lite',
      maxHands: 1
    });
    estado.runtimeModelo = 'tfjs';
    return detector;
  }
}

function actualizarChipModelo(estadoChip, texto) {
  dom.chipModelo.dataset.estado = estadoChip;
  dom.estadoModelo.textContent = texto;
}

function mostrarCapa({ tipo = 'info', icono = '✋', titulo = '', texto = '', boton = null, accion = null, deshabilitado = false }) {
  dom.capaCamara.hidden = false;
  dom.capaCamara.dataset.tipo = tipo;
  dom.capaIcono.textContent = icono;
  dom.capaTitulo.textContent = titulo;
  dom.capaTexto.textContent = texto;

  if (boton) {
    dom.capaBoton.hidden = false;
    dom.capaBoton.textContent = boton;
    dom.capaBoton.disabled = Boolean(deshabilitado);
    dom.capaBoton.onclick = accion || null;
  } else {
    dom.capaBoton.hidden = true;
    dom.capaBoton.onclick = null;
  }
}

function ocultarCapa() {
  dom.capaCamara.hidden = true;
}

function mostrarCapaInicio() {
  if (estado.camaraActiva) {
    ocultarCapa();
    return;
  }

  if (!estado.modeloListo) return;

  mostrarCapa({
    tipo: 'info',
    icono: '✋',
    titulo: 'Activa tu cámara',
    texto: 'Necesitamos acceso a la cámara para detectar tu mano. El video no se graba ni sale de tu dispositivo.',
    boton: 'Activar cámara',
    accion: () => activarCamara()
  });
}

async function activarCamara(deviceId) {
  if (!estado.modeloListo || !estado.detector) {
    mostrarToast('El modelo aún no está listo. Espera un momento o pulsa Reintentar.', 'aviso');
    return;
  }

  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== 'function') {
    manejarErrorCamara({ name: 'SecurityError' });
    return;
  }

  detenerCamara({ silencioso: true });

  mostrarCapa({
    tipo: 'info',
    icono: '📷',
    titulo: 'Solicitando permiso',
    texto: 'Acepta el acceso a la cámara en el navegador. Si no aparece el aviso, revisa el icono del candado en la barra de direcciones.',
    boton: 'Esperando…',
    deshabilitado: true
  });

  const id = deviceId !== undefined ? deviceId : (estado.config.camaraId || '');
  const base = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
  let stream = null;

  try {
    const video = id
      ? { deviceId: { exact: id }, ...base }
      : { facingMode: 'user', ...base };
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video });
  } catch (error) {
    if (error && error.name === 'OverconstrainedError' && id) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: 'user', ...base }
        });
      } catch (segundoError) {
        manejarErrorCamara(segundoError);
        return;
      }
    } else {
      manejarErrorCamara(error);
      return;
    }
  }

  estado.stream = stream;
  dom.video.srcObject = stream;
  dom.videoEntrenar.srcObject = stream;

  try {
    await dom.video.play();
  } catch (error) {
    console.warn('No se pudo reproducir el video principal:', error);
  }

  try {
    await dom.videoEntrenar.play();
  } catch (error) {
    console.warn('No se pudo reproducir el video de entrenamiento:', error);
  }

  estado.camaraActiva = true;
  estado.manoUI = null;
  estado.avisoDeteccion = false;
  estado.framesSinMano = 0;

  dom.btnDetenerCamara.hidden = false;
  dom.capaEntrenar.hidden = true;
  dom.avisoSinMano.hidden = true;
  actualizarEstadoCamara('activa', 'Cámara activa');

  ocultarCapa();
  await listarCamaras();
  iniciarBucleDeteccion();

  mostrarToast('Cámara activada. Muestra tu mano.', 'exito');
  anunciar('Cámara activada. Coloca la mano frente a la cámara.');
}

function detenerCamara({ silencioso = false } = {}) {
  if (estado.rafId) {
    cancelAnimationFrame(estado.rafId);
    estado.rafId = null;
  }

  if (estado.stream) {
    estado.stream.getTracks().forEach((pista) => pista.stop());
    estado.stream = null;
  }

  if (dom.video) dom.video.srcObject = null;
  if (dom.videoEntrenar) dom.videoEntrenar.srcObject = null;

  estado.camaraActiva = false;
  estado.mano = false;
  estado.manoUI = null;
  estado.vector = null;
  estado.prediccion = null;
  estado.confianza = 0;
  estado.candidato = '';
  estado.contador = 0;
  estado.esperandoCambio = false;
  estado.bufferPunta = [];
  estado.movimiento = 0;
  estado.framesSinMano = 0;

  detenerCapturaSostenida();
  limpiarCanvas(dom.canvas);
  limpiarCanvas(dom.canvasEntrenar);

  actualizarUIPrediccion(null);
  actualizarUIMano(false);
  actualizarMovimiento(0);
  actualizarProgreso(0, 'Esperando una mano');

  dom.avisoSinMano.hidden = true;
  dom.btnDetenerCamara.hidden = true;
  dom.capaEntrenar.hidden = false;
  dom.textoCapaEntrenar.textContent = 'Activa la cámara para ver el esqueleto mientras capturas.';
  dom.btnCamaraEntrenar.textContent = 'Activar cámara';
  actualizarEstadoCamara('inactiva', 'Cámara inactiva');

  if (estado.modeloListo) mostrarCapaInicio();
  if (!silencioso) mostrarToast('Cámara detenida', 'aviso');
}

async function listarCamaras() {
  if (!navigator.mediaDevices || typeof navigator.mediaDevices.enumerateDevices !== 'function') {
    dom.selectCamara.innerHTML = '<option>No disponible</option>';
    dom.selectCamara.disabled = true;
    return;
  }

  try {
    const dispositivos = (await navigator.mediaDevices.enumerateDevices())
      .filter((dispositivo) => dispositivo.kind === 'videoinput');

    dom.selectCamara.replaceChildren();

    if (!dispositivos.length) {
      const opcion = document.createElement('option');
      opcion.textContent = 'No se encontraron cámaras';
      dom.selectCamara.append(opcion);
      dom.selectCamara.disabled = true;
      return;
    }

    const pistaActiva = estado.stream && estado.stream.getVideoTracks()[0];
    const ajustes = pistaActiva && pistaActiva.getSettings ? pistaActiva.getSettings() : {};
    const idActivo = ajustes.deviceId || estado.config.camaraId;

    for (let indice = 0; indice < dispositivos.length; indice += 1) {
      const dispositivo = dispositivos[indice];
      const opcion = document.createElement('option');
      opcion.value = dispositivo.deviceId;
      opcion.textContent = dispositivo.label || `Cámara ${indice + 1}`;
      dom.selectCamara.append(opcion);
    }

    dom.selectCamara.disabled = dispositivos.length < 2;

    if (idActivo && dispositivos.some((dispositivo) => dispositivo.deviceId === idActivo)) {
      dom.selectCamara.value = idActivo;
    }
  } catch (error) {
    console.warn('No se pudieron enumerar las cámaras:', error);
    dom.selectCamara.replaceChildren();
    const opcion = document.createElement('option');
    opcion.textContent = 'No se pudieron listar las cámaras';
    dom.selectCamara.append(opcion);
    dom.selectCamara.disabled = true;
  }
}

function manejarErrorCamara(error) {
  console.error('Error de cámara:', error);
  estado.camaraActiva = false;
  actualizarEstadoCamara('error', 'Cámara no disponible');
  dom.btnDetenerCamara.hidden = true;
  dom.capaEntrenar.hidden = false;
  dom.textoCapaEntrenar.textContent = 'No se pudo activar la cámara. Revisa el permiso e inténtalo de nuevo.';
  dom.btnCamaraEntrenar.textContent = 'Reintentar';

  const nombre = error && error.name ? error.name : '';
  let icono = '⚠️';
  let titulo = 'No pudimos abrir la cámara';
  let texto = 'Comprueba que la cámara esté conectada y que otra aplicación no la esté usando.';

  if (nombre === 'NotAllowedError' || nombre === 'PermissionDeniedError' || nombre === 'SecurityError') {
    icono = '🔒';
    titulo = 'Permiso de cámara denegado';
    texto = 'El navegador bloqueó la cámara. Pulsa el candado de la barra de direcciones, permite la cámara y vuelve a intentarlo. La app necesita HTTPS o localhost.';
  } else if (nombre === 'NotFoundError' || nombre === 'DevicesNotFoundError') {
    icono = '📷';
    titulo = 'No se encontró una cámara';
    texto = 'Conecta una cámara web o revisa la configuración de tu dispositivo y vuelve a intentarlo.';
  } else if (nombre === 'NotReadableError' || nombre === 'TrackStartError') {
    icono = '⚠️';
    titulo = 'La cámara está ocupada';
    texto = 'Otra aplicación o pestaña está usando la cámara. Ciérrala y pulsa Reintentar.';
  } else if (nombre === 'OverconstrainedError') {
    icono = '📷';
    titulo = 'Cámara no disponible';
    texto = 'La cámara seleccionada ya no está conectada. Elige otra cámara o pulsa Reintentar.';
  }

  mostrarCapa({
    tipo: 'error',
    icono,
    titulo,
    texto,
    boton: 'Reintentar',
    accion: () => activarCamara()
  });

  mostrarToast(titulo, 'error', 5500);
}

function actualizarEstadoCamara(estadoCamara, texto) {
  dom.estadoCamara.dataset.estado = estadoCamara;
  dom.textoEstadoCamara.textContent = texto;
}

function iniciarBucleDeteccion() {
  if (estado.rafId) cancelAnimationFrame(estado.rafId);

  const iterar = async () => {
    if (!estado.camaraActiva) return;

    const enVistaDeCamara = estado.tabActiva === 'reconocer' || estado.tabActiva === 'entrenar';

    if (enVistaDeCamara && estado.detector) {
      const fuente = estado.tabActiva === 'entrenar' ? dom.videoEntrenar : dom.video;

      if (fuente && fuente.readyState >= 2 && fuente.videoWidth > 0) {
        try {
          const manos = await estado.detector.estimateHands(fuente, { flipHorizontal: false });
          await procesarManos(manos);
        } catch (error) {
          if (!estado.avisoDeteccion) {
            estado.avisoDeteccion = true;
            console.error('Error al procesar el video:', error);
            mostrarToast('Error al procesar el video. Detén y vuelve a activar la cámara.', 'error', 5500);
          }
        }
      }
    }

    if (estado.camaraActiva) {
      estado.rafId = requestAnimationFrame(iterar);
    }
  };

  estado.rafId = requestAnimationFrame(iterar);
}

/* ============================================================
   5. NORMALIZACIÓN, DIBUJO Y CLASIFICACIÓN
   ============================================================ */

function limitar(valor, minimo, maximo) {
  return Math.min(maximo, Math.max(minimo, valor));
}

function redondear(valor, decimales = 4) {
  const factor = 10 ** decimales;
  return Math.round(valor * factor) / factor;
}

function distancia(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Convierte 21 landmarks (x, y) en un vector de 42 valores invariante a
 * traslación y escala:
 * - resta la muñeca (origen)
 * - divide por el tamaño de la palma (muñeca → nudillo del medio)
 * - si la mano es izquierda, refleja X para compensar lateralidad
 */
function normalizarLandmarks(kps, handedness) {
  const puntos = kps.slice(0, 21).map((punto) => ({
    x: Number(punto.x) || 0,
    y: Number(punto.y) || 0
  }));

  const muneca = puntos[0];
  const palma = Math.max(
    distancia(puntos[0], puntos[9]),
    distancia(puntos[0], puntos[5]),
    distancia(puntos[0], puntos[17]),
    0.000001
  );

  const esIzquierda = String(handedness || '').toLowerCase() === 'left';
  const vector = [];

  for (const punto of puntos) {
    let x = (punto.x - muneca.x) / palma;
    const y = (punto.y - muneca.y) / palma;
    if (esIzquierda) x = -x;
    vector.push(redondear(x), redondear(y));
  }

  return vector;
}

/* ------------------------------------------------------------
   Reconocimiento general por geometría de la mano.
   No usa muestras: describe la pose con 21 valores normalizados
   (rectitud/extensión de cada dedo, separación, posición del pulgar,
   orientación y movimiento) y la compara con plantillas del alfabeto.
   El entrenamiento personal se mezcla después como un ajuste fino.
   ------------------------------------------------------------ */

const DEDOS_MANO = Object.freeze({
  indice: Object.freeze([5, 6, 7, 8]),
  medio: Object.freeze([9, 10, 11, 12]),
  anular: Object.freeze([13, 14, 15, 16]),
  menique: Object.freeze([17, 18, 19, 20])
});

const ORDEN_DEDOS = Object.freeze(['indice', 'medio', 'anular', 'menique']);
const PARES_SEPARACION = Object.freeze([
  ['indice', 'medio'],
  ['medio', 'anular'],
  ['anular', 'menique']
]);

const ESTADOS_DEDO = Object.freeze({
  EXTENDIDO: Object.freeze({ rectitud: 0.94, extension: 0.92 }),
  SEMI: Object.freeze({ rectitud: 0.72, extension: 0.66 }),
  GANCHO: Object.freeze({ rectitud: 0.62, extension: 0.55 }),
  CURVO: Object.freeze({ rectitud: 0.52, extension: 0.32 }),
  CERRADO: Object.freeze({ rectitud: 0.45, extension: 0.16 })
});

const ESTADOS_SEPARACION = Object.freeze({
  JUNTOS: 0.06,
  MEDIO: 0.34,
  SEPARADOS: 0.75,
  CRUZADOS: 0.02
});

/*
 * Posiciones típicas del pulgar. x/y se expresan en el marco de la palma:
 * x positivo hacia el lado del índice, y positivo hacia los nudillos.
 */
const ESTADOS_PULGAR = Object.freeze({
  LADO: Object.freeze({ rectitud: 0.85, alcance: 0.68, dIndice: 0.55, dMedio: 0.85, dAnular: 1, dMenique: 1.1, x: 0.95, y: 0.7 }),
  CRUZADO: Object.freeze({ rectitud: 0.75, alcance: 0.8, dIndice: 0.35, dMedio: 0.3, dAnular: 0.35, dMenique: 0.5, x: 0.15, y: 0.65 }),
  FRENTE: Object.freeze({ rectitud: 0.7, alcance: 0.62, dIndice: 0.28, dMedio: 0.3, dAnular: 0.42, dMenique: 0.6, x: 0.45, y: 0.6 }),
  DEBAJO_CUATRO: Object.freeze({ rectitud: 0.5, alcance: 0.45, dIndice: 0.25, dMedio: 0.2, dAnular: 0.2, dMenique: 0.3, x: 0.05, y: 0.32 }),
  DEBAJO_TRES: Object.freeze({ rectitud: 0.55, alcance: 0.5, dIndice: 0.3, dMedio: 0.25, dAnular: 0.25, dMenique: 0.5, x: 0.15, y: 0.4 }),
  DEBAJO_DOS: Object.freeze({ rectitud: 0.55, alcance: 0.55, dIndice: 0.25, dMedio: 0.25, dAnular: 0.55, dMenique: 0.7, x: 0.3, y: 0.45 }),
  TOCA_INDICE: Object.freeze({ rectitud: 0.75, alcance: 1, dIndice: 0.12, dMedio: 0.55, dAnular: 0.75, dMenique: 0.95, x: 0.6, y: 1.15 }),
  TOCA_TODOS: Object.freeze({ rectitud: 0.6, alcance: 0.95, dIndice: 0.12, dMedio: 0.12, dAnular: 0.15, dMenique: 0.22, x: 0.4, y: 1.05 }),
  ENTRE_DOS: Object.freeze({ rectitud: 0.8, alcance: 0.75, dIndice: 0.3, dMedio: 0.3, dAnular: 0.7, dMenique: 0.9, x: 0.35, y: 0.95 }),
  EXTENDIDO: Object.freeze({ rectitud: 0.95, alcance: 0.95, dIndice: 1, dMedio: 1.1, dAnular: 1.2, dMenique: 1.25, x: 1, y: 0.9 }),
  CURVO: Object.freeze({ rectitud: 0.65, alcance: 0.7, dIndice: 0.4, dMedio: 0.5, dAnular: 0.6, dMenique: 0.7, x: 0.7, y: 0.85 }),
  RECOGIDO: Object.freeze({ rectitud: 0.55, alcance: 0.55, dIndice: 0.5, dMedio: 0.6, dAnular: 0.7, dMenique: 0.8, x: 0.6, y: 0.55 })
});

/* Plantillas del alfabeto. El orden de dedos es índice, medio, anular, meñique. */
const PLANTILLAS_FORMA = Object.freeze({
  A: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'LADO', orientacion: 0.05 }),
  B: Object.freeze({ dedos: ['EXTENDIDO', 'EXTENDIDO', 'EXTENDIDO', 'EXTENDIDO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'CRUZADO', orientacion: 0.05 }),
  C: Object.freeze({ dedos: ['SEMI', 'SEMI', 'SEMI', 'SEMI'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'CURVO', orientacion: 0.1 }),
  D: Object.freeze({ dedos: ['EXTENDIDO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'TOCA_INDICE', orientacion: 0.05 }),
  E: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'DEBAJO_CUATRO', orientacion: 0.05 }),
  F: Object.freeze({ dedos: ['GANCHO', 'EXTENDIDO', 'EXTENDIDO', 'EXTENDIDO'], separacion: ['MEDIO', 'JUNTOS', 'JUNTOS'], pulgar: 'TOCA_INDICE', orientacion: 0.05 }),
  G: Object.freeze({ dedos: ['EXTENDIDO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'EXTENDIDO', orientacion: 0.5 }),
  H: Object.freeze({ dedos: ['EXTENDIDO', 'EXTENDIDO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'RECOGIDO', orientacion: 0.5 }),
  I: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'EXTENDIDO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'RECOGIDO', orientacion: 0.05 }),
  J: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'EXTENDIDO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'RECOGIDO', orientacion: 0.05, movimiento: 1 }),
  K: Object.freeze({ dedos: ['EXTENDIDO', 'EXTENDIDO', 'CERRADO', 'CERRADO'], separacion: ['SEPARADOS', 'JUNTOS', 'JUNTOS'], pulgar: 'ENTRE_DOS', orientacion: 0.05 }),
  L: Object.freeze({ dedos: ['EXTENDIDO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'EXTENDIDO', orientacion: 0.05 }),
  M: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'DEBAJO_TRES', orientacion: 0.05 }),
  N: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'DEBAJO_DOS', orientacion: 0.05 }),
  O: Object.freeze({ dedos: ['SEMI', 'SEMI', 'SEMI', 'SEMI'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'TOCA_TODOS', orientacion: 0.05 }),
  P: Object.freeze({ dedos: ['EXTENDIDO', 'EXTENDIDO', 'CERRADO', 'CERRADO'], separacion: ['SEPARADOS', 'JUNTOS', 'JUNTOS'], pulgar: 'ENTRE_DOS', orientacion: 1 }),
  Q: Object.freeze({ dedos: ['EXTENDIDO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'EXTENDIDO', orientacion: 1 }),
  R: Object.freeze({ dedos: ['EXTENDIDO', 'EXTENDIDO', 'CERRADO', 'CERRADO'], separacion: ['CRUZADOS', 'JUNTOS', 'JUNTOS'], pulgar: 'RECOGIDO', orientacion: 0.05 }),
  S: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'FRENTE', orientacion: 0.05 }),
  T: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'ENTRE_DOS', orientacion: 0.05 }),
  U: Object.freeze({ dedos: ['EXTENDIDO', 'EXTENDIDO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'RECOGIDO', orientacion: 0.05 }),
  V: Object.freeze({ dedos: ['EXTENDIDO', 'EXTENDIDO', 'CERRADO', 'CERRADO'], separacion: ['SEPARADOS', 'JUNTOS', 'JUNTOS'], pulgar: 'RECOGIDO', orientacion: 0.05 }),
  W: Object.freeze({ dedos: ['EXTENDIDO', 'EXTENDIDO', 'EXTENDIDO', 'CERRADO'], separacion: ['SEPARADOS', 'SEPARADOS', 'JUNTOS'], pulgar: 'RECOGIDO', orientacion: 0.05 }),
  X: Object.freeze({ dedos: ['GANCHO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'RECOGIDO', orientacion: 0.05 }),
  Y: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'EXTENDIDO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'EXTENDIDO', orientacion: 0.05 }),
  Z: Object.freeze({ dedos: ['EXTENDIDO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'RECOGIDO', orientacion: 0.05, movimiento: 1 }),
  Ñ: Object.freeze({ dedos: ['CERRADO', 'CERRADO', 'CERRADO', 'CERRADO'], separacion: ['JUNTOS', 'JUNTOS', 'JUNTOS'], pulgar: 'DEBAJO_DOS', orientacion: 0.05, movimiento: 1 })
});

const PESOS_CARACTERISTICAS = Object.freeze([
  2.4, 2.4, 2.4, 2.4,       /* rectitud de los cuatro dedos */
  2, 2, 2, 2,               /* extensión de los cuatro dedos */
  1.5, 1.5, 1.5,            /* separación entre dedos */
  1.4, 1.2,                 /* rectitud y alcance del pulgar */
  1, 1, 1, 1,               /* distancia pulgar → cada dedo */
  2.2, 2.2,                 /* posición x/y del pulgar en la palma */
  0.8,                      /* orientación de la mano */
  1                         /* movimiento */
]);

function crearPrototipoForma(plantilla) {
  const vector = new Float32Array(21);

  for (let indice = 0; indice < ORDEN_DEDOS.length; indice += 1) {
    const estadoDedo = ESTADOS_DEDO[plantilla.dedos[indice]] || ESTADOS_DEDO.CURVO;
    vector[indice] = estadoDedo.rectitud;
    vector[4 + indice] = estadoDedo.extension;
  }

  for (let indice = 0; indice < PARES_SEPARACION.length; indice += 1) {
    vector[8 + indice] = ESTADOS_SEPARACION[plantilla.separacion[indice]] || ESTADOS_SEPARACION.JUNTOS;
  }

  const pulgar = ESTADOS_PULGAR[plantilla.pulgar] || ESTADOS_PULGAR.RECOGIDO;
  vector[11] = pulgar.rectitud;
  vector[12] = pulgar.alcance;
  vector[13] = pulgar.dIndice;
  vector[14] = pulgar.dMedio;
  vector[15] = pulgar.dAnular;
  vector[16] = pulgar.dMenique;
  vector[17] = limitar((pulgar.x + 0.5) / 2, 0, 1);
  vector[18] = limitar(pulgar.y / 1.8, 0, 1);
  vector[19] = limitar(plantilla.orientacion || 0, 0, 1);
  vector[20] = limitar(plantilla.movimiento || 0, 0, 1);

  return vector;
}

const PROTOTIPOS_FORMA = (() => {
  const salida = {};
  for (const [letra, plantilla] of Object.entries(PLANTILLAS_FORMA)) {
    salida[letra] = crearPrototipoForma(plantilla);
  }
  return Object.freeze(salida);
})();

function puntoDeVector(vector, indice) {
  return {
    x: vector[indice * 2],
    y: vector[indice * 2 + 1]
  };
}

function normalizarVector2D(punto) {
  const longitud = Math.hypot(punto.x, punto.y) || 1;
  return { x: punto.x / longitud, y: punto.y / longitud };
}

function productoPunto(a, b) {
  return a.x * b.x + a.y * b.y;
}

function rectitudCadena(puntos, indices) {
  let cadena = 0;
  for (let indice = 0; indice < indices.length - 1; indice += 1) {
    cadena += distancia(puntos[indices[indice]], puntos[indices[indice + 1]]);
  }
  if (!cadena) return 0;
  return limitar(distancia(puntos[indices[0]], puntos[indices[indices.length - 1]]) / cadena, 0, 1);
}

function extensionDedo(puntos, indices, ejeLargo) {
  const base = puntos[indices[0]];
  const punta = puntos[indices[indices.length - 1]];
  const haciaPunta = normalizarVector2D({ x: punta.x - base.x, y: punta.y - base.y });
  return limitar((productoPunto(haciaPunta, ejeLargo) + 1) / 2, 0, 1);
}

function separacionDedos(puntos, dedoA, dedoB) {
  const idsA = DEDOS_MANO[dedoA];
  const idsB = DEDOS_MANO[dedoB];
  const baseA = puntos[idsA[0]];
  const baseB = puntos[idsB[0]];
  const puntaA = puntos[idsA[idsA.length - 1]];
  const puntaB = puntos[idsB[idsB.length - 1]];
  const direccionA = normalizarVector2D({ x: puntaA.x - baseA.x, y: puntaA.y - baseA.y });
  const direccionB = normalizarVector2D({ x: puntaB.x - baseB.x, y: puntaB.y - baseB.y });
  const angulo = Math.acos(limitar(productoPunto(direccionA, direccionB), -1, 1)) * (180 / Math.PI);
  const separacionAngulo = limitar(angulo / 32, 0, 1);
  const separacionDistancia = limitar((distancia(puntaA, puntaB) - 0.12) / 0.55, 0, 1);
  return limitar((separacionAngulo + separacionDistancia) / 2, 0, 1);
}

/**
 * Convierte el vector normalizado de 42 valores en 21 características
 * interpretables. Al trabajar sobre el vector ya normalizado, el resultado
 * no depende de la posición, el tamaño ni la lateralidad de la mano.
 */
function extraerCaracteristicas(vector, movimiento = 0) {
  const puntos = [];
  for (let indice = 0; indice < 21; indice += 1) {
    puntos.push(puntoDeVector(vector, indice));
  }

  const muneca = puntos[0];
  const escala = Math.max(
    distancia(muneca, puntos[9]),
    distancia(muneca, puntos[5]),
    distancia(muneca, puntos[17]),
    0.000001
  );

  /* Eje largo de la palma: de la muñeca hacia el nudillo del dedo medio. */
  const ejeLargo = normalizarVector2D({ x: puntos[9].x - muneca.x, y: puntos[9].y - muneca.y });
  const caracteristicas = new Float32Array(21);

  for (let indice = 0; indice < ORDEN_DEDOS.length; indice += 1) {
    const nombre = ORDEN_DEDOS[indice];
    const ids = DEDOS_MANO[nombre];
    caracteristicas[indice] = rectitudCadena(puntos, ids);
    caracteristicas[4 + indice] = extensionDedo(puntos, ids, ejeLargo);
  }

  for (let indice = 0; indice < PARES_SEPARACION.length; indice += 1) {
    const [dedoA, dedoB] = PARES_SEPARACION[indice];
    caracteristicas[8 + indice] = separacionDedos(puntos, dedoA, dedoB);
  }

  caracteristicas[11] = rectitudCadena(puntos, [1, 2, 3, 4]);
  caracteristicas[12] = limitar((distancia(muneca, puntos[4]) / escala - 0.55) / 0.75, 0, 1);
  caracteristicas[13] = limitar(distancia(puntos[4], puntos[8]) / escala / 1.15, 0, 1);
  caracteristicas[14] = limitar(distancia(puntos[4], puntos[12]) / escala / 1.15, 0, 1);
  caracteristicas[15] = limitar(distancia(puntos[4], puntos[16]) / escala / 1.15, 0, 1);
  caracteristicas[16] = limitar(distancia(puntos[4], puntos[20]) / escala / 1.15, 0, 1);

  /* Marco local de la palma: x hacia el índice, y hacia los nudillos. */
  const ejeCorto = normalizarVector2D({ x: puntos[5].x - puntos[17].x, y: puntos[5].y - puntos[17].y });
  const pulgarRelativo = { x: puntos[4].x - muneca.x, y: puntos[4].y - muneca.y };
  const pulgarX = productoPunto(pulgarRelativo, ejeCorto) / escala;
  const pulgarY = productoPunto(pulgarRelativo, ejeLargo) / escala;
  caracteristicas[17] = limitar((pulgarX + 0.5) / 2, 0, 1);
  caracteristicas[18] = limitar(pulgarY / 1.8, 0, 1);

  /* 0 = dedos hacia arriba, 0.5 = de lado, 1 = hacia abajo. */
  const anguloMano = Math.abs(Math.atan2(ejeLargo.x, -ejeLargo.y)) * (180 / Math.PI);
  caracteristicas[19] = limitar(anguloMano / 180, 0, 1);
  caracteristicas[20] = limitar(movimiento / UMBRAL_MOVIMIENTO, 0, 1);

  return caracteristicas;
}

function puntuarCaracteristicas(caracteristicas) {
  const letras = ALFABETOS[estado.config.alfabeto].letras;
  const puntajes = new Map();

  for (const letra of letras) {
    const prototipo = PROTOTIPOS_FORMA[letra];
    if (!prototipo) continue;

    let error = 0;
    let pesoTotal = 0;

    for (let indice = 0; indice < caracteristicas.length; indice += 1) {
      const diferencia = caracteristicas[indice] - prototipo[indice];
      const peso = PESOS_CARACTERISTICAS[indice];
      error += diferencia * diferencia * peso;
      pesoTotal += peso;
    }

    const errorMedio = pesoTotal ? error / pesoTotal : 1;
    puntajes.set(letra, Math.exp(-errorMedio / 0.18));
  }

  return puntajes;
}

function puntuarFormas(vector, movimiento) {
  return puntuarCaracteristicas(extraerCaracteristicas(vector, movimiento));
}

function colorConfianza(confianza) {
  const valor = limitar(Number(confianza) || 0, 0, 1);
  const tono = Math.round(8 + valor * 120);
  return `hsl(${tono} 92% 62%)`;
}

function limpiarCanvas(canvas) {
  if (!canvas) return;
  const contexto = canvas.getContext('2d');
  if (contexto) contexto.clearRect(0, 0, canvas.width, canvas.height);
}

function dibujarMano(canvas, kps, confianza, buffer) {
  if (!canvas) return;

  const ancho = (dom.video && dom.video.videoWidth) ||
    (dom.videoEntrenar && dom.videoEntrenar.videoWidth) || 640;
  const alto = (dom.video && dom.video.videoHeight) ||
    (dom.videoEntrenar && dom.videoEntrenar.videoHeight) || 480;

  if (canvas.width !== ancho || canvas.height !== alto) {
    canvas.width = ancho;
    canvas.height = alto;
  }

  const contexto = canvas.getContext('2d');
  if (!contexto) return;

  contexto.clearRect(0, 0, ancho, alto);

  /* Rastro de la punta del índice (útil para J, Z y Ñ) */
  if (buffer && buffer.length > 2) {
    contexto.save();
    contexto.lineCap = 'round';
    contexto.lineJoin = 'round';
    contexto.lineWidth = Math.max(2, ancho / 220);

    for (let indice = 1; indice < buffer.length; indice += 1) {
      const alfa = (indice / buffer.length) * 0.55;
      contexto.strokeStyle = `rgba(244, 114, 182, ${alfa.toFixed(3)})`;
      contexto.beginPath();
      contexto.moveTo(buffer[indice - 1].x, buffer[indice - 1].y);
      contexto.lineTo(buffer[indice].x, buffer[indice].y);
      contexto.stroke();
    }

    contexto.restore();
  }

  const tienePrediccion = Boolean(estado.prediccion && estado.prediccion.letra);
  const esDesconocida = Boolean(estado.prediccion && estado.prediccion.desconocida);
  const color = tienePrediccion
    ? colorConfianza(confianza)
    : esDesconocida
      ? 'rgba(251, 113, 133, 0.92)'
      : 'rgba(34, 211, 238, 0.92)';
  const grosor = Math.max(2, ancho / 280);

  contexto.save();
  contexto.lineCap = 'round';
  contexto.lineJoin = 'round';
  contexto.strokeStyle = color;
  contexto.shadowColor = color;
  contexto.shadowBlur = Math.max(8, ancho / 80);
  contexto.lineWidth = grosor;

  for (const [a, b] of CONEXIONES) {
    const puntoA = kps[a];
    const puntoB = kps[b];
    if (!puntoA || !puntoB) continue;
    contexto.beginPath();
    contexto.moveTo(puntoA.x, puntoA.y);
    contexto.lineTo(puntoB.x, puntoB.y);
    contexto.stroke();
  }

  contexto.shadowBlur = 0;
  const radioBase = Math.max(2.5, ancho / 190);

  for (let indice = 0; indice < kps.length; indice += 1) {
    const punto = kps[indice];
    if (!punto) continue;
    const radio = PUNTAS.has(indice) ? radioBase * 1.5 : radioBase;
    contexto.beginPath();
    contexto.arc(punto.x, punto.y, radio, 0, Math.PI * 2);
    contexto.fillStyle = indice === 0 ? '#f472b6' : color;
    contexto.fill();
  }

  contexto.restore();
}

async function procesarManos(manos) {
  if (!estado.camaraActiva) return;

  if (!manos || !manos.length) {
    estado.mano = false;
    estado.vector = null;
    estado.prediccion = null;
    estado.confianza = 0;
    estado.candidato = '';
    estado.contador = 0;
    estado.bufferPunta = [];
    estado.movimiento = 0;
    estado.framesSinMano += 1;

    if (estado.framesSinMano > 4) {
      estado.esperandoCambio = false;
    }

    limpiarCanvas(dom.canvas);
    limpiarCanvas(dom.canvasEntrenar);
    actualizarUIPrediccion(null);
    actualizarUIMano(false);
    actualizarMovimiento(0);
    actualizarProgreso(0);
    return;
  }

  const mano = manos[0];
  const kps = mano && mano.keypoints;

  if (!Array.isArray(kps) || kps.length < 21) return;

  estado.mano = true;
  estado.framesSinMano = 0;
  estado.handedness = mano.handedness || '';

  const vector = normalizarLandmarks(kps, estado.handedness);
  estado.vector = vector;

  const palma = Math.max(distancia(kps[0], kps[9]), 1);
  const punta = kps[8];

  if (punta) {
    estado.bufferPunta.push({ x: punta.x, y: punta.y });
    if (estado.bufferPunta.length > VENTANA_MOVIMIENTO) {
      estado.bufferPunta.shift();
    }
  }

  estado.movimiento = calcularMovimiento(estado.bufferPunta, palma);

  const prediccion = await predecir(vector);
  if (!estado.camaraActiva) return;
  estado.prediccion = prediccion;
  estado.confianza = prediccion ? prediccion.confianza : 0;

  if (estado.esperandoCambio && prediccion && prediccion.letra && prediccion.letra !== estado.ultimaLetra) {
    estado.esperandoCambio = false;
  }

  dibujarMano(dom.canvas, kps, estado.confianza, estado.bufferPunta);
  if (estado.tabActiva === 'entrenar') {
    dibujarMano(dom.canvasEntrenar, kps, estado.confianza, estado.bufferPunta);
  }

  actualizarUIPrediccion(prediccion);
  actualizarUIMano(true);
  actualizarMovimiento(estado.movimiento);

  if (estado.tabActiva === 'reconocer') {
    if (prediccion && prediccion.letra) {
      evaluarEstabilizacion(prediccion);
    } else if (estado.esperandoCambio) {
      estado.candidato = '';
      estado.contador = 0;
      actualizarProgreso(0, 'Cambia de seña para continuar');
    } else if (prediccion && prediccion.desconocida) {
      estado.candidato = '';
      estado.contador = 0;
      actualizarProgreso(0, 'Seña no reconocida. Ajusta la pose o entrena esta letra.');
    } else {
      actualizarProgreso(0, 'Muestra una letra del alfabeto');
    }
  }
}

function calcularMovimiento(buffer, palma) {
  if (!buffer || buffer.length < 6 || !palma) return 0;

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const punto of buffer) {
    if (punto.x < minX) minX = punto.x;
    if (punto.x > maxX) maxX = punto.x;
    if (punto.y < minY) minY = punto.y;
    if (punto.y > maxY) maxY = punto.y;
  }

  const amplitud = Math.max(maxX - minX, maxY - minY) / palma;
  return Math.min(3, redondear(amplitud, 3));
}

function calcularConfianzaPrediccion(puntaje, segundoPuntaje) {
  const separacion = puntaje > 0 ? limitar((puntaje - segundoPuntaje) / puntaje, 0, 1) : 0;
  const calidad = 0.85 * puntaje + 0.15 * separacion;
  return limitar(0.45 + 0.55 * calidad, 0, 1);
}

/**
 * Vecinos personales: para cada letra entrenada calcula la distancia media
 * de sus K muestras más cercanas. Una letra sin muestras cercanas recibe
 * similitud 0, así que no puede "robar" una seña que no le corresponde.
 */
function predecirPersonalizado(vector) {
  const porLetra = new Map();
  const muestras = estado.muestrasClasificador;

  if (!muestras.length) return porLetra;

  const k = Math.max(1, Math.min(Math.round(estado.config.k), 15));
  const distanciasPorLetra = new Map();

  for (const muestra of muestras) {
    const valores = muestra.vector;
    let suma = 0;

    for (let indice = 0; indice < 42; indice += 1) {
      const diferencia = valores[indice] - vector[indice];
      suma += diferencia * diferencia;
    }

    let lista = distanciasPorLetra.get(muestra.letra);
    if (!lista) {
      lista = [];
      distanciasPorLetra.set(muestra.letra, lista);
    }

    if (lista.length >= k && suma >= lista[lista.length - 1]) continue;

    let posicion = lista.length;
    while (posicion > 0 && lista[posicion - 1] > suma) posicion -= 1;
    lista.splice(posicion, 0, suma);
    if (lista.length > k) lista.pop();
  }

  for (const [letra, distancias] of distanciasPorLetra) {
    if (!distancias.length) continue;
    const promedio = Math.sqrt(distancias.reduce((suma, valor) => suma + valor, 0) / distancias.length);
    const similitud = Math.exp(-((promedio / RADIO_SIMILITUD_PERSONAL) ** 2));
    porLetra.set(letra, { similitud, distancia: promedio });
  }

  return porLetra;
}

async function predecir(vector) {
  if (!Array.isArray(vector) || vector.length !== 42) return null;

  const formas = puntuarFormas(vector, estado.movimiento);
  const personales = predecirPersonalizado(vector);
  const letras = ALFABETOS[estado.config.alfabeto].letras;
  const combinados = [];

  for (const letra of letras) {
    const puntajeForma = formas.get(letra) || 0;
    const infoPersonal = personales.get(letra);
    const conteo = estado.conteoLetras.get(letra) || 0;

    const pesoPersonal = infoPersonal
      ? Math.min(
        MAX_PESO_PERSONAL,
        PESO_PERSONAL_INICIAL + PESO_PERSONAL_COMPLETO * Math.min(1, conteo / MIN_MUESTRAS)
      )
      : 0;

    const puntajePersonal = infoPersonal ? infoPersonal.similitud : 0;
    const puntaje = (1 - pesoPersonal) * puntajeForma + pesoPersonal * puntajePersonal;

    combinados.push({
      letra,
      puntaje,
      puntajeForma,
      puntajePersonal,
      pesoPersonal
    });
  }

  combinados.sort((a, b) => b.puntaje - a.puntaje);

  const mejor = combinados[0];
  const segundo = combinados[1] || { puntaje: 0 };
  const margen = mejor ? mejor.puntaje - segundo.puntaje : 0;

  /* Si ninguna plantilla encaja, o dos letras están empatadas, no adivinamos.
     Una letra entrenada tampoco puede ganar solo por una similitud personal débil. */
  const soporteSuficiente = Boolean(
    mejor && (mejor.puntajeForma >= UMBRAL_FORMA_MINIMA || mejor.puntajePersonal >= 0.55)
  );

  if (!mejor || !soporteSuficiente || mejor.puntaje < UMBRAL_FORMA_MINIMA || margen < MARGEN_MINIMO) {
    return {
      letra: '',
      desconocida: true,
      confianza: 0,
      puntaje: mejor ? mejor.puntaje : 0,
      margen
    };
  }

  const origen = mejor.pesoPersonal > 0 && mejor.puntajePersonal >= mejor.puntajeForma
    ? 'personal'
    : 'base';

  return {
    letra: mejor.letra,
    confianza: calcularConfianzaPrediccion(mejor.puntaje, segundo.puntaje),
    origen,
    puntaje: mejor.puntaje,
    margen
  };
}

function actualizarConteoLetras() {
  const conteo = new Map();
  for (const muestra of estado.muestrasClasificador) {
    conteo.set(muestra.letra, (conteo.get(muestra.letra) || 0) + 1);
  }
  estado.conteoLetras = conteo;
}

function agregarMuestraClasificador(letra, vector) {
  if (!esVector(vector)) return;
  estado.muestrasClasificador.push({ letra, vector: Float32Array.from(vector) });
  estado.conteoLetras.set(letra, (estado.conteoLetras.get(letra) || 0) + 1);
}

function reconstruirClasificador() {
  estado.muestrasClasificador = [];

  const datos = estado.dataset[estado.config.alfabeto] || {};

  for (const [letra, muestras] of Object.entries(datos)) {
    if (!Array.isArray(muestras)) continue;

    for (const muestra of muestras) {
      if (!esVector(muestra)) continue;
      estado.muestrasClasificador.push({ letra, vector: Float32Array.from(muestra) });
    }
  }

  actualizarConteoLetras();
}

/* ============================================================
   6. ESTABILIZACIÓN Y PALABRA
   ============================================================ */

function evaluarEstabilizacion(prediccion) {
  const { umbral, frames } = estado.config;

  if (estado.esperandoCambio) {
    estado.candidato = '';
    estado.contador = 0;
    actualizarProgreso(0, 'Cambia de seña para continuar');
    return;
  }

  if (!prediccion || !Number.isFinite(prediccion.confianza) || prediccion.confianza < umbral) {
    estado.candidato = '';
    estado.contador = 0;
    actualizarProgreso(0, prediccion ? 'Confianza insuficiente' : 'Esperando una seña estable');
    return;
  }

  const letra = prediccion.letra;

  if (letra === estado.candidato) {
    estado.contador += 1;
  } else {
    estado.candidato = letra;
    estado.contador = 1;
  }

  /* J, Z y Ñ necesitan trayectoria: sin movimiento no se aceptan. */
  if (LETRAS_MOVIMIENTO.has(letra) && estado.config.exigirMovimiento && estado.movimiento < UMBRAL_MOVIMIENTO) {
    if (estado.contador >= frames) estado.contador = frames - 1;
    actualizarProgreso(estado.contador / frames, `Haz el movimiento de ${letra}`);
    return;
  }

  actualizarProgreso(estado.contador / frames);

  if (estado.contador >= frames) {
    aceptarLetra(letra, prediccion.confianza);
  }
}

function aceptarLetra(letra, confianza) {
  estado.ultimaLetra = letra;
  estado.esperandoCambio = true;
  estado.candidato = '';
  estado.contador = 0;

  animarLetra();
  actualizarProgreso(0, 'Letra aceptada. Cambia de seña para continuar');

  if (estado.practica.activa) {
    evaluarPractica(letra);
  } else {
    agregarLetraPalabra(letra);
    anunciar(`Letra ${letra} aceptada con ${Math.round((confianza || 0) * 100)} por ciento de confianza.`);
  }
}

function animarLetra() {
  dom.letraDetectada.classList.remove('pulso');
  void dom.letraDetectada.offsetWidth;
  dom.letraDetectada.classList.add('pulso');
}

function agregarLetraPalabra(letra) {
  if (estado.palabra.length >= MAX_PALABRA) {
    mostrarToast(`La palabra alcanzó el límite de ${MAX_PALABRA} caracteres`, 'aviso');
    return;
  }

  estado.palabra += letra;
  renderizarPalabra();
}

function agregarEspacio() {
  if (!estado.palabra || estado.palabra.endsWith(' ')) return;
  agregarLetraPalabra(' ');
}

function borrarUltimaLetra() {
  estado.palabra = estado.palabra.slice(0, -1);
  renderizarPalabra();
}

function limpiarPalabra() {
  estado.palabra = '';
  renderizarPalabra();
}

function guardarPalabraActual() {
  const texto = normalizarPalabra(estado.palabra);
  if (!texto) {
    mostrarToast('Forma una palabra antes de guardarla', 'aviso');
    return;
  }

  agregarPalabraDiccionario(texto, true);
  registrarHistorial(texto);
  estado.palabra = '';
  renderizarPalabra();
  renderizarDiccionario();
  renderizarHistorial();
  mostrarToast(`“${texto}” guardada en el diccionario`, 'exito');
  anunciar(`Palabra ${texto} guardada.`);
}

function renderizarPalabra() {
  const texto = estado.palabra;
  dom.palabraActual.textContent = texto || '—';
  dom.palabraActual.classList.toggle('vacia', !texto);

  const vacia = !texto;
  dom.btnBorrarLetra.disabled = vacia;
  dom.btnLimpiarPalabra.disabled = vacia;
  dom.btnGuardarPalabra.disabled = vacia;
  dom.btnPronunciarPalabra.disabled = vacia;

  actualizarSugerencias();
}

/* ============================================================
   7. ENTRENAMIENTO Y DATASET
   ============================================================ */

function totalMuestras(alfabeto = estado.config.alfabeto) {
  const datos = estado.dataset[alfabeto] || {};
  return Object.values(datos).reduce((suma, muestras) => suma + (Array.isArray(muestras) ? muestras.length : 0), 0);
}

function letrasConMuestras(alfabeto = estado.config.alfabeto) {
  const datos = estado.dataset[alfabeto] || {};
  return Object.keys(datos).filter((letra) => Array.isArray(datos[letra]) && datos[letra].length > 0);
}

function renderizarGridLetras() {
  const alfabeto = ALFABETOS[estado.config.alfabeto];
  const datos = estado.dataset[estado.config.alfabeto] || {};
  const fragmento = document.createDocumentFragment();

  for (const letra of alfabeto.letras) {
    const muestras = Array.isArray(datos[letra]) ? datos[letra].length : 0;
    const boton = document.createElement('button');
    boton.type = 'button';
    boton.className = 'letra-btn';
    boton.dataset.letra = letra;
    boton.setAttribute('aria-pressed', String(estado.letraEntrenamiento === letra));
    boton.setAttribute('aria-label', `Letra ${letra}, ${muestras} muestras${muestras > 0 ? '' : ', reconocimiento base'}${LETRAS_MOVIMIENTO.has(letra) ? ', seña con movimiento' : ''}`);

    if (muestras > 0) boton.classList.add('con-muestras');
    if (muestras >= MIN_MUESTRAS) boton.classList.add('entrenada');
    if (estado.letraEntrenamiento === letra) boton.classList.add('seleccionada');

    const simbolo = document.createElement('span');
    simbolo.className = 'letra-btn__simbolo';
    simbolo.textContent = letra;

    const contador = document.createElement('span');
    contador.className = 'letra-btn__muestras';
    contador.textContent = muestras > 0 ? String(muestras) : '·';

    boton.append(simbolo, contador);

    if (LETRAS_MOVIMIENTO.has(letra)) {
      const marca = document.createElement('span');
      marca.className = 'letra-btn__mov';
      marca.textContent = '⚡';
      marca.setAttribute('aria-hidden', 'true');
      boton.append(marca);
    }

    fragmento.append(boton);
  }

  dom.gridLetras.replaceChildren(fragmento);
}

function actualizarBotonLetraUI(letra) {
  const boton = dom.gridLetras.querySelector(`.letra-btn[data-letra="${letra}"]`);
  if (!boton) return;

  const datos = estado.dataset[estado.config.alfabeto] || {};
  const muestras = Array.isArray(datos[letra]) ? datos[letra].length : 0;
  const contador = boton.querySelector('.letra-btn__muestras');

  if (contador) contador.textContent = muestras > 0 ? String(muestras) : '·';
  boton.classList.toggle('con-muestras', muestras > 0);
  boton.classList.toggle('entrenada', muestras >= MIN_MUESTRAS);
  boton.setAttribute('aria-label', `Letra ${letra}, ${muestras} muestras${muestras > 0 ? '' : ', reconocimiento base'}${LETRAS_MOVIMIENTO.has(letra) ? ', seña con movimiento' : ''}`);
}

function seleccionarLetraEntrenamiento(letra) {
  if (!ALFABETOS[estado.config.alfabeto].letras.includes(letra)) return;

  detenerCapturaSostenida();
  estado.letraEntrenamiento = letra;
  renderizarGridLetras();
  actualizarContadoresLetra();
}

function actualizarContadoresLetra() {
  const letra = estado.letraEntrenamiento;

  if (!letra) {
    dom.letraSeleccionada.textContent = '—';
    dom.contadorLetra.textContent = '0 muestras';
    dom.metaLetra.textContent = 'Elige una letra del alfabeto';
    dom.barraLetra.style.width = '0%';
    dom.btnCapturar.disabled = true;
    return;
  }

  const datos = estado.dataset[estado.config.alfabeto] || {};
  const muestras = Array.isArray(datos[letra]) ? datos[letra].length : 0;

  dom.letraSeleccionada.textContent = letra;
  dom.contadorLetra.textContent = `${muestras} ${muestras === 1 ? 'muestra' : 'muestras'}`;

  if (muestras === 0) {
    dom.metaLetra.textContent = 'Sin personalizar · la base general ya la reconoce';
  } else if (muestras < MIN_MUESTRAS) {
    dom.metaLetra.textContent = `Faltan ${MIN_MUESTRAS - muestras} para personalizarla del todo`;
  } else {
    dom.metaLetra.textContent = 'Personalizada · reconocimiento afinado';
  }

  dom.barraLetra.style.width = `${Math.min(100, (muestras / MIN_MUESTRAS) * 100)}%`;
  dom.btnCapturar.disabled = false;
}

function actualizarResumenEntrenamiento() {
  const alfabeto = ALFABETOS[estado.config.alfabeto];
  const datos = estado.dataset[estado.config.alfabeto] || {};

  let conMuestras = 0;
  let listas = 0;
  let pendientes = 0;

  for (const letra of alfabeto.letras) {
    const cantidad = Array.isArray(datos[letra]) ? datos[letra].length : 0;
    if (cantidad > 0) conMuestras += 1;
    if (cantidad >= MIN_MUESTRAS) listas += 1;
    if (cantidad === 0) pendientes += 1;
  }

  dom.numConMuestras.textContent = String(conMuestras);
  dom.numListas.textContent = String(listas);
  dom.numPendientes.textContent = String(pendientes);
  dom.alfabetoEntrenar.textContent = `${alfabeto.nombre} · ${alfabeto.letras.length} letras`;
}

function animarCaptura() {
  dom.btnCapturar.classList.remove('capturada');
  void dom.btnCapturar.offsetWidth;
  dom.btnCapturar.classList.add('capturada');
}

function capturarMuestra(silencioso = false) {
  if (!estado.camaraActiva) {
    if (!silencioso) mostrarToast('Activa la cámara para capturar muestras', 'aviso');
    return;
  }

  if (!estado.mano || !estado.vector || estado.vector.length !== 42) {
    if (!silencioso) mostrarToast('No se detecta la mano. Acércala a la cámara y prueba de nuevo.', 'aviso', 2200);
    return;
  }

  if (!estado.letraEntrenamiento) {
    if (!silencioso) mostrarToast('Elige una letra antes de capturar', 'aviso');
    return;
  }

  const ahora = performance.now();
  if (ahora - estado.ultimaCaptura < 100) return;
  estado.ultimaCaptura = ahora;

  const letra = estado.letraEntrenamiento;
  const datos = estado.dataset[estado.config.alfabeto];
  const lista = datos[letra] || (datos[letra] = []);

  if (lista.length >= MAX_MUESTRAS) {
    if (!silencioso) mostrarToast(`Límite de ${MAX_MUESTRAS} muestras por letra alcanzado`, 'aviso');
    return;
  }

  const vector = estado.vector.slice();
  lista.push(vector);
  agregarMuestraClasificador(letra, vector);

  guardarDataset();
  actualizarBotonLetraUI(letra);
  actualizarContadoresLetra();
  actualizarResumenEntrenamiento();
  animarCaptura();

  if (lista.length === MIN_MUESTRAS) {
    mostrarToast(`Letra ${letra} personalizada con ${MIN_MUESTRAS} muestras`, 'exito', 2600);
    anunciar(`Letra ${letra} personalizada con ${MIN_MUESTRAS} muestras.`);
  }
}

function iniciarCapturaSostenida() {
  if (estado.capturando) return;

  if (!estado.letraEntrenamiento) {
    mostrarToast('Elige una letra antes de capturar', 'aviso');
    return;
  }

  if (!estado.camaraActiva) {
    mostrarToast('Activa la cámara para capturar muestras', 'aviso');
    return;
  }

  estado.capturando = true;
  capturarMuestra();
  estado.intervaloCaptura = window.setInterval(() => capturarMuestra(true), 170);
}

function detenerCapturaSostenida() {
  estado.capturando = false;
  if (estado.intervaloCaptura) {
    clearInterval(estado.intervaloCaptura);
    estado.intervaloCaptura = null;
  }
}

async function borrarLetraActual() {
  const letra = estado.letraEntrenamiento;

  if (!letra) {
    mostrarToast('Selecciona una letra primero', 'aviso');
    return;
  }

  const datos = estado.dataset[estado.config.alfabeto] || {};
  const cantidad = Array.isArray(datos[letra]) ? datos[letra].length : 0;

  if (!cantidad) {
    mostrarToast(`La letra ${letra} no tiene muestras`, 'aviso');
    return;
  }

  const confirmado = await confirmar(
    `Borrar la letra ${letra}`,
    `Se eliminarán ${cantidad} muestras de ${letra}. Esta acción no se puede deshacer.`,
    'Borrar letra'
  );

  if (!confirmado) return;

  delete estado.dataset[estado.config.alfabeto][letra];
  reconstruirClasificador();
  guardarDataset();
  renderizarGridLetras();
  actualizarContadoresLetra();
  actualizarResumenEntrenamiento();
  actualizarUIPrediccion(null);
  resetEstabilizacion();

  if (estado.practica.activa) {
    if (!letrasConMuestras().length) {
      detenerPractica();
    } else if (!letrasConMuestras().includes(estado.practica.objetivo)) {
      nuevaLetraPractica();
    }
  }

  mostrarToast(`Letra ${letra} eliminada`, 'aviso');
}

async function borrarTodo() {
  const alfabeto = ALFABETOS[estado.config.alfabeto];
  const cantidad = totalMuestras();

  if (!cantidad) {
    mostrarToast('No hay muestras para borrar', 'aviso');
    return;
  }

  const confirmado = await confirmar(
    `Borrar todo el dataset de ${alfabeto.nombre}`,
    `Se eliminarán ${cantidad} muestras de ${alfabeto.letras.length} letras. Exporta un respaldo antes si lo necesitas. Esta acción no se puede deshacer.`,
    'Borrar todo'
  );

  if (!confirmado) return;

  estado.dataset[estado.config.alfabeto] = {};
  reconstruirClasificador();
  guardarDataset();
  renderizarGridLetras();
  actualizarContadoresLetra();
  actualizarResumenEntrenamiento();
  actualizarUIPrediccion(null);
  resetEstabilizacion();
  detenerPractica();
  mostrarToast(`Dataset de ${alfabeto.nombre} borrado`, 'aviso');
}

function exportarDataset() {
  const paquete = {
    app: 'senas',
    version: 1,
    exportado: new Date().toISOString(),
    alfabeto: estado.config.alfabeto,
    datos: estado.dataset
  };

  const blob = new Blob([JSON.stringify(paquete, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const enlace = document.createElement('a');
  enlace.href = url;
  enlace.download = `senas-dataset-${new Date().toISOString().slice(0, 10)}.json`;
  enlace.style.display = 'none';
  document.body.append(enlace);
  enlace.click();
  enlace.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1500);

  mostrarToast('Dataset exportado como JSON', 'exito');
}

function sanearDataset(crudo) {
  const resultado = { lsm: {}, asl: {} };
  let total = 0;

  if (!crudo || typeof crudo !== 'object') return { resultado, total };

  for (const alfabeto of ['lsm', 'asl']) {
    const fuente = crudo[alfabeto];
    if (!fuente || typeof fuente !== 'object') continue;

    for (const letra of ALFABETOS[alfabeto].letras) {
      const muestras = fuente[letra];
      if (!Array.isArray(muestras)) continue;

      const validas = muestras.filter(esVector).slice(0, MAX_MUESTRAS);
      if (validas.length) {
        resultado[alfabeto][letra] = validas;
        total += validas.length;
      }
    }
  }

  return { resultado, total };
}

async function importarDataset(archivo) {
  try {
    const texto = await archivo.text();
    const crudo = JSON.parse(texto);
    const fuente = crudo && typeof crudo === 'object' && crudo.datos ? crudo.datos : crudo;
    const { resultado, total } = sanearDataset(fuente);

    if (!total) {
      mostrarToast('El archivo no contiene muestras válidas de 42 valores', 'error', 5000);
      return;
    }

    let agregadas = 0;

    for (const alfabeto of ['lsm', 'asl']) {
      for (const [letra, muestras] of Object.entries(resultado[alfabeto])) {
        const destino = estado.dataset[alfabeto][letra] || (estado.dataset[alfabeto][letra] = []);
        const antes = destino.length;

        for (const muestra of muestras) {
          if (destino.length >= MAX_MUESTRAS) break;
          destino.push(muestra);
        }

        agregadas += destino.length - antes;
      }
    }

    reconstruirClasificador();
    guardarDataset();
    renderizarGridLetras();
    actualizarContadoresLetra();
    actualizarResumenEntrenamiento();
    actualizarUIPrediccion(null);
    resetEstabilizacion();

    mostrarToast(`Importación completada: ${agregadas} muestras nuevas`, 'exito', 4500);
  } catch (error) {
    console.error('Error al importar:', error);
    mostrarToast('No se pudo leer el archivo JSON. Revisa el formato.', 'error', 5000);
  }
}

/* ============================================================
   8. DICCIONARIO, HISTORIAL, VOZ Y PRÁCTICA
   ============================================================ */

function normalizarPalabra(texto) {
  return String(texto || '')
    .trim()
    .replace(/\s+/g, ' ')
    .toLocaleUpperCase('es-MX');
}

function agregarPalabraDiccionario(texto, incrementarUso) {
  const palabra = normalizarPalabra(texto);
  if (!palabra) return false;

  const actual = estado.diccionario[palabra];

  if (actual) {
    if (incrementarUso) actual.usos = (actual.usos || 0) + 1;
    actual.ultima = Date.now();
  } else {
    estado.diccionario[palabra] = {
      usos: incrementarUso ? 1 : 0,
      ultima: Date.now()
    };
  }

  guardarDiccionario();
  return true;
}

function registrarHistorial(texto) {
  const limpio = normalizarPalabra(texto);
  if (!limpio) return;

  estado.historial.unshift({ texto: limpio, fecha: Date.now() });
  estado.historial = estado.historial.slice(0, MAX_HISTORIAL);
  guardarHistorial();
}

function formatearFecha(milisegundos) {
  const fecha = new Date(milisegundos);
  if (Number.isNaN(fecha.getTime())) return '';
  return fecha.toLocaleString('es-MX', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function crearBotonIcono(icono, etiqueta, accion, peligro = false) {
  const boton = document.createElement('button');
  boton.type = 'button';
  boton.className = `btn-icono${peligro ? ' peligro' : ''}`;
  boton.textContent = icono;
  boton.title = etiqueta;
  boton.setAttribute('aria-label', etiqueta);
  boton.addEventListener('click', (evento) => {
    Promise.resolve(accion(evento)).catch((error) => console.error(error));
  });
  return boton;
}

function renderizarDiccionario() {
  const entradas = Object.entries(estado.diccionario)
    .filter(([palabra, datos]) => palabra && datos && typeof datos === 'object')
    .sort((a, b) => (b[1].usos || 0) - (a[1].usos || 0) || a[0].localeCompare(b[0], 'es'));

  dom.contadorDiccionario.textContent = `${entradas.length} ${entradas.length === 1 ? 'palabra' : 'palabras'}`;
  dom.listaDiccionario.replaceChildren();

  if (!entradas.length) {
    dom.vacioDiccionario.hidden = false;
    return;
  }

  dom.vacioDiccionario.hidden = true;
  const fragmento = document.createDocumentFragment();

  for (const [palabra, datos] of entradas) {
    const elemento = document.createElement('li');
    elemento.className = 'item';

    const texto = document.createElement('span');
    texto.className = 'item__texto';
    texto.textContent = palabra;

    const meta = document.createElement('span');
    meta.className = 'item__meta';
    meta.textContent = `${datos.usos || 0} ${datos.usos === 1 ? 'uso' : 'usos'}`;
    texto.append(meta);

    const acciones = document.createElement('span');
    acciones.className = 'item__acciones';

    acciones.append(
      crearBotonIcono('🔊', `Pronunciar ${palabra}`, () => pronunciar(palabra)),
      crearBotonIcono('↺', `Usar ${palabra} en la palabra actual`, () => {
        estado.palabra = palabra;
        renderizarPalabra();
        cambiarTab('reconocer');
      }),
      crearBotonIcono('🗑', `Eliminar ${palabra} del diccionario`, async () => {
        const confirmado = await confirmar(
          `Eliminar “${palabra}”`,
          'La palabra saldrá del diccionario. El historial de frases no cambia.',
          'Eliminar'
        );
        if (!confirmado) return;

        delete estado.diccionario[palabra];
        guardarDiccionario();
        renderizarDiccionario();
        actualizarSugerencias();
        mostrarToast('Palabra eliminada del diccionario', 'aviso');
      }, true)
    );

    elemento.append(texto, acciones);
    fragmento.append(elemento);
  }

  dom.listaDiccionario.append(fragmento);
}

function renderizarHistorial() {
  dom.listaHistorial.replaceChildren();

  if (!estado.historial.length) {
    dom.vacioHistorial.hidden = false;
    dom.btnLimpiarHistorial.disabled = true;
    return;
  }

  dom.vacioHistorial.hidden = true;
  dom.btnLimpiarHistorial.disabled = false;

  const fragmento = document.createDocumentFragment();

  for (const entrada of estado.historial) {
    const elemento = document.createElement('li');
    elemento.className = 'item';

    const texto = document.createElement('span');
    texto.className = 'item__texto';
    texto.textContent = entrada.texto;

    const meta = document.createElement('span');
    meta.className = 'item__meta';
    meta.textContent = formatearFecha(entrada.fecha);
    texto.append(meta);

    const acciones = document.createElement('span');
    acciones.className = 'item__acciones';

    acciones.append(
      crearBotonIcono('🔊', `Pronunciar ${entrada.texto}`, () => pronunciar(entrada.texto)),
      crearBotonIcono('↺', `Reutilizar ${entrada.texto}`, () => {
        estado.palabra = entrada.texto;
        renderizarPalabra();
        cambiarTab('reconocer');
      }),
      crearBotonIcono('🗑', `Eliminar ${entrada.texto} del historial`, () => {
        const indice = estado.historial.indexOf(entrada);
        if (indice >= 0) {
          estado.historial.splice(indice, 1);
          guardarHistorial();
          renderizarHistorial();
        }
      }, true)
    );

    elemento.append(texto, acciones);
    fragmento.append(elemento);
  }

  dom.listaHistorial.append(fragmento);
}

function actualizarSugerencias() {
  dom.sugerencias.replaceChildren();
  const prefijo = estado.palabra.trim();

  if (!prefijo) {
    dom.sugerencias.hidden = true;
    return;
  }

  const entradas = Object.entries(estado.diccionario)
    .filter(([palabra]) => palabra.startsWith(prefijo) && palabra !== prefijo)
    .sort((a, b) => (b[1].usos || 0) - (a[1].usos || 0) || a[0].localeCompare(b[0], 'es'))
    .slice(0, 6);

  if (!entradas.length) {
    dom.sugerencias.hidden = true;
    return;
  }

  dom.sugerencias.hidden = false;
  const fragmento = document.createDocumentFragment();

  for (const [palabra, datos] of entradas) {
    const boton = document.createElement('button');
    boton.type = 'button';
    boton.className = 'sugerencia';
    boton.dataset.palabra = palabra;
    boton.setAttribute('aria-label', `Completar ${palabra}, ${datos.usos || 0} usos`);

    boton.append(document.createTextNode(palabra));

    const usos = document.createElement('span');
    usos.textContent = String(datos.usos || 0);
    boton.append(usos);

    fragmento.append(boton);
  }

  dom.sugerencias.append(fragmento);
}

function pronunciar(texto) {
  const limpio = String(texto || estado.palabra || '').trim();

  if (!limpio) {
    mostrarToast('Forma o elige una palabra para pronunciarla', 'aviso');
    return;
  }

  if (!('speechSynthesis' in window)) {
    mostrarToast('Tu navegador no soporta síntesis de voz', 'error');
    return;
  }

  window.speechSynthesis.cancel();

  const locucion = new SpeechSynthesisUtterance(limpio);
  locucion.lang = 'es-MX';
  locucion.rate = 0.95;
  locucion.pitch = 1;

  const voces = window.speechSynthesis.getVoices();
  const voz = voces.find((vozDisponible) => vozDisponible.lang.toLowerCase() === 'es-mx') ||
    voces.find((vozDisponible) => vozDisponible.lang.toLowerCase().startsWith('es'));

  if (voz) locucion.voice = voz;

  window.speechSynthesis.speak(locucion);
  mostrarToast(`Pronunciando “${limpio}”`, 'exito', 1800);
}

function actualizarUIPracticaStats() {
  dom.practicaPuntos.textContent = String(estado.practica.puntos);
  dom.practicaRacha.textContent = String(estado.practica.racha);
  dom.practicaMejor.textContent = String(estado.practica.mejor);
}

function iniciarPractica() {
  if (estado.practica.activa) {
    detenerPractica();
    return;
  }

  const disponibles = letrasConMuestras();

  if (!disponibles.length) {
    mostrarToast('Entrena al menos una letra antes de practicar', 'aviso', 4000);
    cambiarTab('entrenar');
    return;
  }

  estado.practica = {
    activa: true,
    objetivo: '',
    puntos: 0,
    racha: 0,
    mejor: 0
  };

  dom.practicaPanel.classList.add('activa');
  dom.practicaEstado.textContent = 'Activo';
  dom.btnPractica.textContent = 'Terminar práctica';
  dom.btnPracticaSaltar.disabled = false;
  dom.practicaFeedback.className = 'practica__feedback';

  nuevaLetraPractica();
  dom.practicaFeedback.textContent = 'Muestra la letra indicada. Las letras de práctica no se añaden a la palabra.';
  anunciar('Modo práctica iniciado.');
}

function detenerPractica() {
  if (!estado.practica.activa) return;

  estado.practica.activa = false;
  estado.practica.objetivo = '';
  dom.practicaPanel.classList.remove('activa');
  dom.practicaEstado.textContent = 'Inactivo';
  dom.btnPractica.textContent = 'Iniciar práctica';
  dom.btnPracticaSaltar.disabled = true;
  dom.practicaObjetivo.textContent = '?';
  dom.practicaFeedback.className = 'practica__feedback';
  dom.practicaFeedback.textContent = `Práctica terminada: ${estado.practica.puntos} puntos y mejor racha de ${estado.practica.mejor}.`;
  actualizarUIPracticaStats();
  anunciar('Modo práctica terminado.');
}

function nuevaLetraPractica() {
  const disponibles = letrasConMuestras();

  if (!disponibles.length) {
    detenerPractica();
    return;
  }

  let letra = disponibles[Math.floor(Math.random() * disponibles.length)];

  if (disponibles.length > 1) {
    let intentos = 0;
    while (letra === estado.practica.objetivo && intentos < 12) {
      letra = disponibles[Math.floor(Math.random() * disponibles.length)];
      intentos += 1;
    }
  }

  estado.practica.objetivo = letra;
  dom.practicaObjetivo.textContent = letra;
  actualizarUIPracticaStats();
  anunciar(`Muestra la letra ${letra}.`);
}

function evaluarPractica(letra) {
  if (!estado.practica.activa) return;

  const objetivo = estado.practica.objetivo;

  if (letra === objetivo) {
    estado.practica.puntos += 10;
    estado.practica.racha += 1;
    estado.practica.mejor = Math.max(estado.practica.mejor, estado.practica.racha);
    dom.practicaFeedback.className = 'practica__feedback exito';
    dom.practicaFeedback.textContent = `¡Correcto! +10 puntos. Racha de ${estado.practica.racha}.`;
    anunciar(`Correcto. Era la letra ${letra}.`);
    nuevaLetraPractica();
  } else {
    estado.practica.racha = 0;
    estado.practica.puntos = Math.max(0, estado.practica.puntos - 2);
    dom.practicaFeedback.className = 'practica__feedback error';
    dom.practicaFeedback.textContent = `Detecté ${letra}; la meta era ${objetivo}. Sigue intentando.`;
    anunciar(`Incorrecto. La meta era ${objetivo}.`);
    actualizarUIPracticaStats();
  }
}

/* ============================================================
   9. INTERFAZ, ACCESIBILIDAD Y ARRANQUE
   ============================================================ */

function actualizarUIPrediccion(prediccion) {
  const letra = prediccion && prediccion.letra ? prediccion.letra : '?';
  const desconocida = Boolean(prediccion && prediccion.desconocida);

  if (dom.letraDetectada.textContent !== letra) {
    dom.letraDetectada.textContent = letra;
  }

  dom.letraDetectada.classList.toggle('desconocida', desconocida);

  const porcentaje = prediccion
    ? Math.round(limitar(prediccion.confianza, 0, 1) * 100)
    : 0;

  dom.confianzaValor.textContent = `${porcentaje}%`;
  dom.barraConfianza.style.width = `${porcentaje}%`;

  const umbral = Math.round(estado.config.umbral * 100);
  dom.confianzaValor.style.color = !prediccion || desconocida
    ? 'var(--texto-suave)'
    : porcentaje >= umbral
      ? '#a5f3fc'
      : porcentaje >= 50
        ? '#fde68a'
        : '#fecdd3';

  if (dom.origenPrediccion) {
    let origen = '';
    if (prediccion && prediccion.letra) {
      origen = prediccion.origen === 'personal' ? 'personal' : 'base';
      dom.origenPrediccion.textContent = origen === 'personal' ? 'Personalizada' : 'Base general';
    } else {
      dom.origenPrediccion.textContent = desconocida ? 'Sin coincidencia' : '—';
      origen = desconocida ? 'desconocida' : '';
    }
    dom.origenPrediccion.dataset.origen = origen;
  }
}

function actualizarUIMano(detectada) {
  const cambio = estado.manoUI !== detectada;
  estado.manoUI = detectada;

  if (cambio) {
    dom.estadoMano.dataset.estado = detectada ? 'ok' : 'espera';
    dom.textoEstadoMano.textContent = detectada ? 'Mano detectada' : 'Sin mano';
    dom.avisoSinMano.hidden = detectada || !estado.camaraActiva;
  }

  dom.estadoLateralidad.textContent = detectada ? nombreLateralidad(estado.handedness) : '—';
}

function nombreLateralidad(handedness) {
  const valor = String(handedness || '').toLowerCase();
  if (valor === 'left') return 'Mano izquierda';
  if (valor === 'right') return 'Mano derecha';
  return '—';
}

function actualizarMovimiento(valor) {
  const movimiento = Number.isFinite(valor) ? Math.max(0, valor) : 0;
  dom.valorMovimiento.textContent = movimiento.toFixed(2);
  dom.barraMovimiento.style.width = `${Math.min(100, (movimiento / 0.8) * 100)}%`;
}

function actualizarProgreso(progreso, mensaje) {
  const valor = limitar(Number(progreso) || 0, 0, 1);
  const porcentaje = Math.round(valor * 100);
  const frames = estado.config.frames;

  dom.barraEstabilizacion.style.width = `${porcentaje}%`;
  dom.progresoEstabilizacion.setAttribute('aria-valuenow', String(porcentaje));
  dom.contadorEstabilizacion.textContent = `${Math.min(estado.contador, frames)}/${frames}`;

  if (mensaje) {
    dom.textoEstabilizacion.textContent = mensaje;
  } else if (!estado.mano) {
    dom.textoEstabilizacion.textContent = 'Esperando una mano';
  } else if (valor <= 0) {
    dom.textoEstabilizacion.textContent = 'Esperando una seña estable';
  } else {
    dom.textoEstabilizacion.textContent = 'Mantén la seña';
  }
}

function resetEstabilizacion() {
  estado.candidato = '';
  estado.contador = 0;
  estado.esperandoCambio = false;
  actualizarProgreso(0);
}

function mostrarToast(mensaje, tipo = 'info', duracion = 3200) {
  if (!dom.toasts) return;

  while (dom.toasts.children.length >= 4) {
    dom.toasts.firstElementChild.remove();
  }

  const toast = document.createElement('div');
  toast.className = `toast ${tipo}`;
  toast.textContent = mensaje;
  dom.toasts.append(toast);

  window.setTimeout(() => {
    toast.classList.add('saliendo');
    window.setTimeout(() => toast.remove(), 320);
  }, duracion);
}

function anunciar(texto) {
  if (!dom.anuncio) return;
  dom.anuncio.textContent = '';
  window.setTimeout(() => {
    dom.anuncio.textContent = texto;
  }, 40);
}

function confirmar(titulo, mensaje, textoBoton = 'Eliminar') {
  return new Promise((resolver) => {
    if (!dom.dialogoConfirmar || typeof dom.dialogoConfirmar.showModal !== 'function') {
      resolver(window.confirm(`${titulo}\n\n${mensaje}`));
      return;
    }

    dom.dialogoTitulo.textContent = titulo;
    dom.dialogoMensaje.textContent = mensaje;
    dom.dialogoBtnConfirmar.textContent = textoBoton;
    dom.dialogoConfirmar.returnValue = 'cancelar';

    const alCerrar = () => {
      dom.dialogoConfirmar.removeEventListener('close', alCerrar);
      resolver(dom.dialogoConfirmar.returnValue === 'confirmar');
    };

    dom.dialogoConfirmar.addEventListener('close', alCerrar);
    dom.dialogoConfirmar.showModal();
  });
}

function cambiarTab(vista) {
  if (!vista) return;

  estado.tabActiva = vista;

  for (const tab of dom.tabs) {
    const activa = tab.dataset.vista === vista;
    tab.classList.toggle('activa', activa);
    tab.setAttribute('aria-selected', String(activa));
    tab.tabIndex = activa ? 0 : -1;
  }

  for (const seccion of dom.vistas) {
    seccion.hidden = seccion.id !== `vista-${vista}`;
  }

  resetEstabilizacion();

  if (vista === 'entrenar') {
    actualizarMiniCamara();
    renderizarGridLetras();
    actualizarContadoresLetra();
    actualizarResumenEntrenamiento();
    if (estado.camaraActiva) dom.videoEntrenar.play().catch(() => {});
  }

  if (vista === 'reconocer') {
    renderizarPalabra();
    if (estado.camaraActiva) dom.video.play().catch(() => {});
  }

  if (vista === 'diccionario') {
    renderizarDiccionario();
    renderizarHistorial();
  }

  if (vista === 'ayuda') {
    renderizarReferencia();
  }
}

function manejarTeclasTabs(evento) {
  const teclas = ['ArrowRight', 'ArrowLeft', 'Home', 'End'];
  if (!teclas.includes(evento.key)) return;

  evento.preventDefault();
  const indice = dom.tabs.indexOf(evento.currentTarget);
  let nuevoIndice = indice;

  if (evento.key === 'ArrowRight') nuevoIndice = (indice + 1) % dom.tabs.length;
  if (evento.key === 'ArrowLeft') nuevoIndice = (indice - 1 + dom.tabs.length) % dom.tabs.length;
  if (evento.key === 'Home') nuevoIndice = 0;
  if (evento.key === 'End') nuevoIndice = dom.tabs.length - 1;

  const tab = dom.tabs[nuevoIndice];
  if (!tab) return;
  cambiarTab(tab.dataset.vista);
  tab.focus();
}

function cambiarAlfabeto(nuevo) {
  if (!ALFABETOS[nuevo]) {
    dom.selectAlfabeto.value = estado.config.alfabeto;
    return;
  }

  if (nuevo === estado.config.alfabeto) return;

  estado.config.alfabeto = nuevo;
  guardarConfig();

  estado.letraEntrenamiento = null;
  estado.candidato = '';
  estado.contador = 0;
  estado.ultimaLetra = '';
  estado.esperandoCambio = false;
  estado.framesSinMano = 0;

  if (estado.practica.activa) detenerPractica();

  reconstruirClasificador();
  renderizarGridLetras();
  actualizarContadoresLetra();
  actualizarResumenEntrenamiento();
  renderizarReferencia();
  actualizarUIPrediccion(null);
  resetEstabilizacion();

  mostrarToast(`Alfabeto ${ALFABETOS[nuevo].nombre} activo`, 'exito');
  anunciar(`Alfabeto ${ALFABETOS[nuevo].nombre} activo.`);
}

function renderizarReferencia() {
  const alfabeto = ALFABETOS[estado.config.alfabeto];
  dom.nombreAlfabetoAyuda.textContent = alfabeto.nombre;
  dom.referenciaAlfabeto.replaceChildren();

  const fragmento = document.createDocumentFragment();

  for (const letra of alfabeto.letras) {
    const boton = document.createElement('button');
    boton.type = 'button';
    boton.className = 'sena-card';
    boton.setAttribute('aria-label', `Entrenar la letra ${letra}: ${alfabeto.senas[letra]}`);

    const simbolo = document.createElement('span');
    simbolo.className = 'sena-card__letra';
    simbolo.textContent = letra;

    const cuerpo = document.createElement('span');
    cuerpo.className = 'sena-card__cuerpo';

    const descripcion = document.createElement('span');
    descripcion.className = 'sena-card__desc';
    descripcion.textContent = alfabeto.senas[letra];
    cuerpo.append(descripcion);

    if (LETRAS_MOVIMIENTO.has(letra)) {
      const etiqueta = document.createElement('span');
      etiqueta.className = 'sena-card__tag';
      etiqueta.textContent = '⚡ movimiento · experimental';
      cuerpo.append(etiqueta);
    }

    boton.append(simbolo, cuerpo);
    boton.addEventListener('click', () => {
      seleccionarLetraEntrenamiento(letra);
      cambiarTab('entrenar');
      if (!estado.camaraActiva) {
        mostrarToast('Activa la cámara en Entrenar para capturar muestras', 'aviso');
      }
    });

    fragmento.append(boton);
  }

  dom.referenciaAlfabeto.append(fragmento);
}

function actualizarMiniCamara() {
  dom.capaEntrenar.hidden = estado.camaraActiva;
}

function aplicarConfiguracionUI() {
  dom.selectAlfabeto.value = estado.config.alfabeto;
  dom.rangoK.value = String(estado.config.k);
  dom.valorK.textContent = String(estado.config.k);
  dom.rangoUmbral.value = String(Math.round(estado.config.umbral * 100));
  dom.valorUmbral.textContent = `${Math.round(estado.config.umbral * 100)}%`;
  dom.rangoFrames.value = String(estado.config.frames);
  dom.valorFrames.textContent = String(estado.config.frames);
  dom.checkMovimiento.checked = estado.config.exigirMovimiento;
}

function configurarUI() {
  /* Pestañas */
  for (const tab of dom.tabs) {
    tab.addEventListener('click', () => cambiarTab(tab.dataset.vista));
    tab.addEventListener('keydown', manejarTeclasTabs);
  }

  /* Alfabeto y cámara */
  dom.selectAlfabeto.addEventListener('change', () => cambiarAlfabeto(dom.selectAlfabeto.value));

  dom.selectCamara.addEventListener('change', () => {
    const id = dom.selectCamara.value;
    if (!id) return;
    estado.config.camaraId = id;
    guardarConfig();
    activarCamara(id);
  });

  dom.btnDetenerCamara.addEventListener('click', () => detenerCamara());
  dom.btnCamaraEntrenar.addEventListener('click', () => activarCamara());

  /* Ajustes */
  dom.rangoK.addEventListener('input', () => {
    estado.config.k = Number(dom.rangoK.value);
    dom.valorK.textContent = String(estado.config.k);
    guardarConfig();
  });

  dom.rangoUmbral.addEventListener('input', () => {
    estado.config.umbral = Number(dom.rangoUmbral.value) / 100;
    dom.valorUmbral.textContent = `${dom.rangoUmbral.value}%`;
    guardarConfig();
  });

  dom.rangoFrames.addEventListener('input', () => {
    estado.config.frames = Number(dom.rangoFrames.value);
    dom.valorFrames.textContent = String(estado.config.frames);
    estado.contador = 0;
    estado.candidato = '';
    actualizarProgreso(0);
    guardarConfig();
  });

  dom.checkMovimiento.addEventListener('change', () => {
    estado.config.exigirMovimiento = dom.checkMovimiento.checked;
    guardarConfig();
  });

  /* Palabra */
  dom.btnEspacio.addEventListener('click', agregarEspacio);
  dom.btnBorrarLetra.addEventListener('click', borrarUltimaLetra);
  dom.btnLimpiarPalabra.addEventListener('click', limpiarPalabra);
  dom.btnGuardarPalabra.addEventListener('click', guardarPalabraActual);
  dom.btnPronunciarPalabra.addEventListener('click', () => pronunciar(estado.palabra));

  dom.sugerencias.addEventListener('click', (evento) => {
    const boton = evento.target.closest('button[data-palabra]');
    if (!boton) return;
    estado.palabra = boton.dataset.palabra;
    renderizarPalabra();
  });

  /* Práctica */
  dom.btnPractica.addEventListener('click', iniciarPractica);
  dom.btnPracticaSaltar.addEventListener('click', () => {
    if (!estado.practica.activa) return;
    nuevaLetraPractica();
    dom.practicaFeedback.className = 'practica__feedback';
    dom.practicaFeedback.textContent = 'Nueva letra. ¡Tú puedes!';
  });

  /* Entrenamiento */
  dom.gridLetras.addEventListener('click', (evento) => {
    const boton = evento.target.closest('button[data-letra]');
    if (!boton) return;
    seleccionarLetraEntrenamiento(boton.dataset.letra);
  });

  dom.btnCapturar.addEventListener('pointerdown', (evento) => {
    if (dom.btnCapturar.disabled) return;
    if (evento.pointerType === 'mouse' && evento.button !== 0) return;
    evento.preventDefault();

    try {
      dom.btnCapturar.setPointerCapture(evento.pointerId);
    } catch (error) {
      /* Algunos navegadores no permiten capturar el puntero; no es crítico. */
    }

    iniciarCapturaSostenida();
  });

  for (const evento of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    dom.btnCapturar.addEventListener(evento, detenerCapturaSostenida);
  }

  dom.btnCapturar.addEventListener('click', (evento) => {
    /* Los clics de teclado o lector de pantalla tienen detail 0. */
    if (evento.detail === 0) capturarMuestra();
  });

  dom.btnCapturar.addEventListener('contextmenu', (evento) => evento.preventDefault());

  dom.btnExportar.addEventListener('click', exportarDataset);
  dom.btnImportar.addEventListener('click', () => dom.inputImportar.click());

  dom.inputImportar.addEventListener('change', async () => {
    const archivo = dom.inputImportar.files && dom.inputImportar.files[0];
    if (archivo) await importarDataset(archivo);
    dom.inputImportar.value = '';
  });

  dom.btnBorrarLetra.addEventListener('click', borrarLetraActual);
  dom.btnBorrarTodo.addEventListener('click', borrarTodo);

  /* Diccionario e historial */
  dom.formDiccionario.addEventListener('submit', (evento) => {
    evento.preventDefault();
    const texto = dom.inputPalabra.value;

    if (!normalizarPalabra(texto)) {
      mostrarToast('Escribe una palabra válida', 'aviso');
      return;
    }

    agregarPalabraDiccionario(texto, false);
    dom.inputPalabra.value = '';
    renderizarDiccionario();
    actualizarSugerencias();
    mostrarToast('Palabra añadida al diccionario', 'exito');
  });

  dom.btnLimpiarHistorial.addEventListener('click', async () => {
    if (!estado.historial.length) return;

    const confirmado = await confirmar(
      'Limpiar el historial',
      'Se eliminarán todas las frases guardadas. El diccionario no cambia.',
      'Limpiar historial'
    );

    if (!confirmado) return;

    estado.historial = [];
    guardarHistorial();
    renderizarHistorial();
    mostrarToast('Historial limpiado', 'aviso');
  });

  /* Ciclo de vida */
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) detenerCapturaSostenida();
  });

  window.addEventListener('beforeunload', volcarDatasetInmediato);
  window.addEventListener('pagehide', volcarDatasetInmediato);

  if ('speechSynthesis' in window) {
    window.speechSynthesis.getVoices();
  }
}

function renderizarTodo() {
  renderizarGridLetras();
  actualizarContadoresLetra();
  actualizarResumenEntrenamiento();
  renderizarPalabra();
  renderizarDiccionario();
  renderizarHistorial();
  renderizarReferencia();
  actualizarUIPracticaStats();
  actualizarMiniCamara();
}

async function iniciar() {
  cachearDOM();
  estado.almacenamientoOk = probarAlmacenamiento();
  cargarDatos();

  dom.btnCamaraEntrenar.disabled = true;
  configurarUI();
  aplicarConfiguracionUI();
  reconstruirClasificador();
  renderizarTodo();

  if (!estado.almacenamientoOk) {
    mostrarToast('El almacenamiento local no está disponible. La app funciona, pero no recordará tus datos.', 'aviso', 6500);
  }

  if (!window.isSecureContext) {
    mostrarToast('La cámara requiere HTTPS o localhost. Si abriste el archivo directamente, usa un servidor local.', 'aviso', 7000);
  }

  await cargarModelo();
}

document.addEventListener('DOMContentLoaded', iniciar);
