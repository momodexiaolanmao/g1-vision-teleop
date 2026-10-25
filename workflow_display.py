"""Large Chinese sampling cues over the camera image; English fallback off Windows."""
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

@lru_cache(maxsize=2)
def _font(size):
    path=Path('C:/Windows/Fonts/msyh.ttc')
    return ImageFont.truetype(str(path),size) if path.is_file() else ImageFont.load_default(size=size)

def render_workflow_banner(frame, text):
    chinese='等待就位：全身入镜，双脚落地，手臂放松'
    numbers=re.findall(r'\d+(?:\.\d+)?',text)
    if text.startswith('GO TO CAMERA:'):
        chinese=f'请走到镜头前，{numbers[0]} 秒后开始检查站姿'
    elif text.startswith('STAND STILL:'):
        chinese='请保持静止，正在自动标定'
    elif text.startswith('CALIBRATED:'):
        chinese=f'标定完成，请保持静止，{numbers[0]} 秒后开始采样'
    elif text.startswith('SAMPLING:'):
        chinese=f'正在采样，请保持静止，还剩 {numbers[0]} 秒'
    elif text.startswith('MOVEMENT SAMPLING:'):
        chinese=f'动作采样：缓慢转移重心、交替提膝，还剩 {numbers[0]} 秒'
    elif text.startswith('SAMPLING FINISHED'):
        chinese='采样结束，可以活动了'
    elif text.startswith('GAIT READY'):
        chinese='步态就绪：缓慢转移重心，弯膝抬脚'
    elif text.startswith('GAIT PREPARING'):
        chinese='已识别迈步，正在准备支撑'
        if 'stance not stable' in text:
            chinese='已识别迈步，等待机器人双脚站稳'
        elif 'upper body' in text:
            chinese='已识别迈步，正在调整腰和肩的姿态'
    elif text.startswith('GAIT RUNNING'):
        chinese='步态运行中；停止抬脚约 1.25 秒后开始收步'
    elif text.startswith('GAIT SETTLING'):
        chinese='正在收步，请保持双脚落地'
    elif text.startswith('GAIT FEET HIDDEN'):
        chinese='脚部识别不足，步态已暂停；请露出脚踝、脚跟、脚尖'
    elif text.startswith('GAIT LEGS UNCLEAR'):
        chinese='腿部识别不足，步态已暂停；请露出髋、膝和双脚'
    elif text.startswith('GAIT INPUT HELD'):
        chinese='人体输入丢失，步态已暂停；请回到镜头前'
    elif text.startswith('GAIT FALLEN'):
        chinese='机器人已倒下；按 R 重置，再按 C 重新标定'
    canvas=Image.fromarray(frame[:,:,::-1].copy())
    draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,canvas.width,80),fill=(16,20,24))
    finished=text.startswith('SAMPLING FINISHED')
    color=(100,245,150) if finished else (255,220,100)
    primary=chinese if Path('C:/Windows/Fonts/msyh.ttc').is_file() else text
    draw.text((12,5),primary,font=_font(30 if canvas.width>=900 else 23),fill=color)
    draw.text((12,48),text,font=_font(17),fill=(230,235,240))
    return np.asarray(canvas)[:,:,::-1].copy()
