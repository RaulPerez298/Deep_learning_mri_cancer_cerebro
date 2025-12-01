#!/bin/bash

# Carpetas
ORIGEN="/media/ral298/dis500g/proyectoDeep/"
DESTINO="/home/ral298/Documentos/proyectoDeep/"

# Archivo de log
LOGFILE="$HOME/sincronizar.log"

# Opciones de rsync
RSYNC_OPCIONES="-a --delete --inplace --whole-file --no-compress --progress --info=stats2"

# OPCIONES OPTIMIZADAS PARA CONTENIDO MIXTO
#RSYNC_OPCIONES="-a --delete --inplace --no-compress --progress --info=stats2 --partial --partial-dir=.rsync-partial --size-only --modify-window=1"

# Mostrar simulación
echo "Simulando sincronización (no se copiará nada todavía)..."
ionice -c2 -n0 nice -n -15 rsync $RSYNC_OPCIONES --dry-run "$ORIGEN" "$DESTINO" | tee "$LOGFILE"



echo ""
read -p "¿Deseas proceder con la sincronización real? (s/n): " RESPUESTA
echo $RESPUESTA
if [[ "$RESPUESTA" == "s" || "$RESPUESTA" == "S" ]]; then
    echo "Iniciando sincronización real..."
    ionice -c2 -n0 nice -n -15 rsync $RSYNC_OPCIONES "$ORIGEN" "$DESTINO" | tee -a "$LOGFILE"
    echo "¡Sincronización completada exitosamente!"
else
    echo "Sincronización cancelada. No se hicieron cambios."
fi

