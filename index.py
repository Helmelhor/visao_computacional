import cv2
import mediapipe as mp
import math  # Para calcular distância entre pontos

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Número máximo de mãos a detectar
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils  # Para desenhar os landmarks

# Inicializa a webcam
cap = cv2.VideoCapture(0)

def distancia(lm1, lm2, w, h):
    # Calcula a distância euclidiana entre dois pontos (landmarks)
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.hypot(x2 - x1, y2 - y1)

# Função para detectar se a mão está "fechada" (polegar e indicador juntos)
def mao_fechada(hand_landmarks, w, h):
    thumb_tip = hand_landmarks.landmark[4]   # Ponta do polegar
    index_tip = hand_landmarks.landmark[8]   # Ponta do indicador
    dist = distancia(thumb_tip, index_tip, w, h)
    if dist < 20:  # Valor pode ser ajustado conforme necessário
        return True
    return False

def polegar_mindinho_juntos(hand_landmarks, w, h):
    thumb_tip = hand_landmarks.landmark[4]    # Ponta do polegar
    pinky_tip = hand_landmarks.landmark[20]  # Ponta do mindinho
    dist = distancia(thumb_tip, pinky_tip, w, h)
    if dist < 40:  # Ajuste o valor conforme necessário
        return True
    return False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erro ao capturar o frame. Saindo...")
        break

    # Espelha o frame horizontalmente (efeito espelho)
    frame = cv2.flip(frame, 1)
    # Converte BGR (OpenCV) para RGB (MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processa o frame e detecta as mãos
    results = hands.process(frame_rgb)

    # Desenha os landmarks e verifica gestos se uma mão for detectada
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            h, w, _ = frame.shape
            # Verifica se a mão está fechada (polegar e indicador juntos)
            if mao_fechada(hand_landmarks, w, h):
                # Ação: exibe texto na tela
                cv2.putText(frame, 'MAO FECHADA', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                # Ação: exibe texto na tela
                cv2.putText(frame, 'MAO ABERTA', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Mostra o frame
    cv2.imshow('deteccao de maos - MediaPipe', frame)
    
    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()

print(hand_landmarks)