TFM - SISTEMA DE AUMENTO HUMANO ASISTIDO POR IA
Material para revisión del tutor

Esta carpeta contiene el estado actual del desarrollo experimental del TFM.

01_NOTEBOOK_Y_CODIGO
- Notebook completo del proyecto.
- Versión Python exportada del notebook.
- PDF del notebook para revisión sin necesidad de ejecución.
- La versión principal y más reciente es FINAL_2.

02_MODELOS
Contiene los modelos entrenados utilizados durante el desarrollo.

Modelos utilizados por el prototipo final:
- pedestrian_light_v1_best.pt
- pedestrian_light_v2_best.pt

Modelo experimental:
- pedestrian_light_v2_refined_best.pt

El modelo V2 refined corresponde a un entrenamiento experimental posterior
realizado para comprobar si podía mejorar el V2 anterior.
Actualmente NO se utiliza en el prototipo final.

También se incluyen los clasificadores CLIP + Gate empleados para determinar
si un semáforo detectado resulta accionable para el peatón.

03_RESULTADOS
Contiene los principales resultados del sistema:
- decisiones finales de riesgo
- evaluación manual
- matriz de confusión
- resúmenes por vídeo
- resultados del subsistema de semáforos
- resultados experimentales

04_PROTOTIPO
Contiene vídeos generados por el sistema:
- prototipo base
- prototipo end-to-end / híbrido

05_VÍDEOS
Contiene los vídeos originales utilizados en las pruebas y evaluación.

06_DOCUMENTACION
Contiene documentación complementaria del proyecto.

- README_REVISION_TUTOR.txt:
  guía breve del contenido de la carpeta de revisión.

- requirements.txt:
  lista de dependencias y versiones del entorno Python 3.12 utilizado para ejecutar el proyecto.

ENTORNO
El desarrollo final se ha ejecutado con Python 3.12.

NOTA
El notebook contiene algunas rutas absolutas correspondientes al equipo
utilizado durante el desarrollo. Para ejecutarlo en otro ordenador sería
necesario adaptar dichas rutas.

Este material se proporciona para revisión del estado actual del TFM.
