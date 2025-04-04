import os
import subprocess
import argparse
from pathlib import Path
import whisper


def convert_audio_to_wav(input_file, output_file):
    """
    Convierte archivos de audio (como ogg de WhatsApp) a formato WAV
    utilizando ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-ar",
        "16000",  # Frecuencia de muestreo recomendada para Whisper
        "-ac",
        "1",  # Mono
        "-c:a",
        "pcm_s16le",  # Formato PCM 16 bits
        output_file,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error al convertir el audio: {e}")
        return False


def transcribe_audio(audio_file, model_size="base"):
    """
    Transcribe el audio utilizando el modelo Whisper
    """
    try:
        # Cargar modelo de Whisper
        model = whisper.load_model(model_size)

        # Realizar la transcripción
        result = model.transcribe(audio_file)

        return result["text"]
    except Exception as e:
        print(f"Error en la transcripción: {e}")
        return None


def process_whatsapp_voice(input_dir, output_dir, model_size="base"):
    """
    Procesa todos los mensajes de voz de WhatsApp en un directorio
    y genera archivos de texto con las transcripciones.
    """
    print("Iniciando el proceso de transcripción de mensajes de voz de WhatsApp...")
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Crear directorio de salida si no existe
    output_path.mkdir(parents=True, exist_ok=True)

    # Extensiones comunes de mensajes de voz de WhatsApp
    voice_extensions = [".opus", ".ogg", ".m4a"]

    # Contador de archivos procesados
    processed = 0

    for file in input_path.iterdir():
        if file.is_file() and file.suffix.lower() in voice_extensions:
            print(f"Procesando: {file.name}")

            # Crear nombre para archivo WAV temporal
            temp_wav = output_path / f"{file.stem}_temp.wav"

            # Convertir a WAV
            if convert_audio_to_wav(str(file), str(temp_wav)):
                # Transcribir audio
                transcript = transcribe_audio(str(temp_wav), model_size)

                if transcript:
                    # Guardar transcripción en archivo de texto
                    text_file = output_path / f"{file.stem}.txt"
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(transcript)

                    print(f"Transcripción guardada en: {text_file}")
                    processed += 1

                # Eliminar archivo WAV temporal
                if temp_wav.exists():
                    os.remove(temp_wav)

    print(f"\nProceso completado. {processed} archivos de audio transcritos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convertir mensajes de voz de WhatsApp a texto"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Directorio con los mensajes de voz"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Directorio donde guardar las transcripciones",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Tamaño del modelo Whisper a utilizar (default: base)",
    )

    args = parser.parse_args()

    process_whatsapp_voice(args.input, args.output, args.model)
