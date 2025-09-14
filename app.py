import gradio as gr
import numpy as np
import cv2
import tempfile
import os
import traceback

# Попытка импортировать PaddleOCR и EasyOCR (если не установлены — будет видно в логах)
PADDLE_AVAILABLE = False
EASY_AVAILABLE = False
paddle_ocr = None
easy_reader = None
paddle_import_error = None
easy_import_error = None

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    paddle_import_error = traceback.format_exc()
    PADDLE_AVAILABLE = False

try:
    import easyocr
    EASY_AVAILABLE = True
except Exception as e:
    easy_import_error = traceback.format_exc()
    EASY_AVAILABLE = False

# Пытаемся инициализировать модели лениво (чтобы показать ошибки сразу в лог)
try:
    if PADDLE_AVAILABLE:
        # создание может занять время — но делаем один раз при старте
        paddle_ocr = PaddleOCR(use_angle_cls=True, lang="ru")
except Exception:
    paddle_ocr = None
    paddle_import_error = traceback.format_exc()
    PADDLE_AVAILABLE = False

try:
    if EASY_AVAILABLE:
        # easyocr.Reader создаётся только если easyocr установлен
        easy_reader = easyocr.Reader(['ru', 'en'])
except Exception:
    easy_reader = None
    easy_import_error = traceback.format_exc()
    EASY_AVAILABLE = False

def ocr_receipt(image):
    """
    Возвращает (recognized_text, debug_log)
    """
    debug_lines = []
    try:
        if image is None:
            return "", "Ошибка: изображение не загружено."

        # Если Gradio даёт PIL.Image, преобразуем в numpy
        if not isinstance(image, np.ndarray):
            # gr.Image(type="pil") даёт PIL.Image
            image = np.array(image)

        # Убедимся, что у нас RGB или RGBA
        if image.ndim == 2:
            # grayscale -> BGR
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # RGBA -> RGB
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            # RGB -> BGR
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Сохраняем временный файл (PaddleOCR удобнее получает путь)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpf:
            tmp_path = tmpf.name
            cv2.imwrite(tmp_path, img_bgr)

        debug_lines.append(f"Временный файл: {tmp_path}")

        # Первый вариант — PaddleOCR
        if paddle_ocr is not None:
            debug_lines.append("Пробуем PaddleOCR...")
            try:
                res = paddle_ocr.ocr(tmp_path, cls=True)  # иногда нужно cls=True
                debug_lines.append("PaddleOCR вернул результат.")
                # Собираем текст из структуры результата (без предположений)
                lines = []
                # res может быть списком страниц или одной страницы
                try:
                    # обычно res = [page], где page = [[box, (text, conf)], ...]
                    for page in res:
                        if isinstance(page, list):
                            for item in page:
                                # item может быть [box, (text, conf)]
                                try:
                                    txt = item[1][0]
                                except Exception:
                                    txt = str(item)
                                lines.append(txt)
                        else:
                            lines.append(str(page))
                except Exception:
                    # fallback
                    debug_lines.append("Неожиданная структура результата PaddleOCR.")
                    lines = [str(res)]

                recognized = "\n".join(lines).strip()
                os.remove(tmp_path)
                if not recognized:
                    debug_lines.append("PaddleOCR вернул пустой текст.")
                return recognized, "\n".join(debug_lines)
            except Exception as e:
                debug_lines.append("Ошибка при запуске PaddleOCR:")
                debug_lines.append(traceback.format_exc())
                # продолжим к fallback

        # Fallback — EasyOCR
        if easy_reader is not None:
            debug_lines.append("Пробуем EasyOCR...")
            try:
                # easyocr может принять numpy или путь; используем numpy
                # reader.readtext возвращает list of [bbox, text, conf] или если detail=0 list of text
                res = easy_reader.readtext(tmp_path, detail=1)  # detail=1 возвращает текст с bbox и conf
                debug_lines.append("EasyOCR вернул результат.")
                lines = []
                for item in res:
                    try:
                        txt = item[1]
                    except Exception:
                        txt = str(item)
                    lines.append(txt)
                recognized = "\n".join(lines).strip()
                os.remove(tmp_path)
                if not recognized:
                    debug_lines.append("EasyOCR вернул пустой текст.")
                return recognized, "\n".join(debug_lines)
            except Exception:
                debug_lines.append("Ошибка при запуске EasyOCR:")
                debug_lines.append(traceback.format_exc())
                try:
                    os.remove(tmp_path)
                except:
                    pass

        # Если ни одна модель не доступна — показываем подсказку
        debug_lines.append("Ни PaddleOCR, ни EasyOCR не доступны или обе вернули ошибку.")
        if not PADDLE_AVAILABLE:
            debug_lines.append("PaddleOCR недоступен. Ошибка импорта/инициализации:")
            debug_lines.append(paddle_import_error or "нет дополнительной информации")
        if not EASY_AVAILABLE:
            debug_lines.append("EasyOCR недоступен. Ошибка импорта/инициализации:")
            debug_lines.append(easy_import_error or "нет дополнительной информации")

        debug_lines.append("\nРекомендации:")
        debug_lines.append("1) Установите paddleocr и paddlepaddle (pip install paddleocr paddlepaddle) или easyocr (pip install easyocr).")
        debug_lines.append("2) Проверьте, что команда запускается в том же окружении, где установлены пакеты.")
        debug_lines.append("3) Если используете Windows и возникает ошибка установки paddlepaddle, попробуйте подходящий wheel с официального сайта paddlepaddle.")

        return "", "\n".join(debug_lines)

    except Exception:
        debug = traceback.format_exc()
        return "", f"Внутренняя ошибка:\n{debug}"
    finally:
        # удаляем временный файл, если он остался
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass

# ========== UI Gradio ==========
with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink")) as demo:
    gr.Markdown("<h2 style='text-align:center;color:#d63384'>🧾 OCR Чек — Диагностический режим</h2>")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Загрузите чек")
            submit_btn = gr.Button("📸 Распознать")
        with gr.Column():
            recognized = gr.Textbox(label="Распознанный текст", lines=20, interactive=False)
            debug = gr.Textbox(label="Лог/диагностика (читай при ошибках)", lines=20, interactive=False)

    submit_btn.click(fn=ocr_receipt, inputs=image_input, outputs=[recognized, debug])

if __name__ == "__main__":
    demo.launch()
