"""Generate a self-contained local editor with all G1 joints and body channels."""
import json
from pathlib import Path
from pose_retarget import CHANNEL_LABELS

ROOT = Path(__file__).absolute().parent


def main():
    for profile in ("unitree_g1", "unitree_g1_physics", "unitree_g1_walking", "unitree_g1_tracking"):
        template = (ROOT / "templates/unitree_g1_editor.html").read_text(encoding="utf-8")
        if profile.endswith(("physics","walking","tracking")):
            template = template.replace("全身影子跟随", "物理站立跟随").replace("实时影子跟随", "物理站立跟随")
            template = template.replace("把你的全身动作，映射到 G1。", "让 G1 站住，再跟随你的动作。")
            template = template.replace("已配置双腿、腰部、双臂和手腕。", "双臂、手腕和腰部跟随你的姿态；双腿优先保持双脚接地和平衡，并随屈膝做有限下蹲。")
            template = template.replace("实际跟随在相机与 MuJoCo 窗口运行。", "预览显示映射参考角度；实际角度还会受平衡、接触和力矩约束。")
            template = template.replace('class="check"', 'class="check" style="display:none"')
            template = template.replace('<label>标定时相机距离', '<label style="display:none">标定时相机距离')
            template = template.replace("导出 unitree_g1_mapping.json", "导出 unitree_g1_physics_mapping.json")
            start = template.index("<p>影子跟随用于预览")
            end = template.index("</p>", start) + 4
            template = template[:start] + "<p>物理模式启用重力、脚底碰撞、摩擦和电机力矩限制。平衡优先：腿部不能照抄任意人体角度，双脚保持原地支撑。当前适合抬手、摆臂、转腰和有限下蹲；尚未实现迈步、转身行走、单脚站立和跳跃。<br>操作：全身入镜后按 C 标定；空格暂停跟随（仍保持平衡）；R 重置仿真；Q 退出。</p>" + template[end:]
        if profile.endswith('walking'):
            template=template.replace('物理站立跟随','物理迈步跟随')
            template=template.replace('让 G1 站住，再跟随你的动作。','随你摆臂、转腰，按抬脚意图尝试迈步。')
            template=template.replace('双臂、手腕和腰部跟随你的姿态；双腿优先保持双脚接地和平衡，并随屈膝做有限下蹲。',
                '抬脚发出迈步请求，上身姿势经过防抖后跟随。支撑准备完成才进入步态；腿部由策略协调，不直接照抄人体腿部关节角。')
            template=template.replace('导出 unitree_g1_physics_mapping.json','导出 unitree_g1_walking_mapping.json')
            start=template.index('<p>物理模式启用重力')
            end=template.index('</p>',start)+4
            template=template[:start]+'<p>同侧画面抬高估计超过约 3.5 厘米、膝盖比标定时多弯曲约 12°并持续至少 80 毫秒，才发出迈步请求。请求保留 1.25 秒，新的抬脚会续期；准备至少 0.65 秒，等腰肩与双脚支撑满足条件后进入步态。停止抬脚后等请求到期再收步；人物丢失、脚腿低可信或暂停立即取消请求。活动期间以约 0.35 米/秒前后往返，距起点前后约 25 厘米换向；短停保留方向，惯性可能越过换向位置。左右脚由策略协调，不能复刻任意舞蹈。<br>双击 start_unitree_g1.cmd 后走入镜头，8 秒就位、站稳自动标定。C 重新就位；N 后 C 重选；空格暂停，R 重置，Q 退出。25 秒脚步复测用 check_unitree_g1_motion.cmd，按提示转移重心、交替提膝，看到“采样结束，可以活动了”并听到提示音后再回电脑。新报告逐物理步骤累计脚底高度和接触力。最新 25 秒现场迈步复测通过，两脚离地约 1.98/1.81 厘米且未跌倒；脚部遮挡仍暂停，不能保证每次抬脚对应；详见 G1_STEPPING_zh.md。TWIST2 仍属实验，真人相机测试曾跌倒，不作默认推荐；详见 G1_TRACKING_zh.md 和 OTHER_PROJECTS_zh.md。</p>'+template[end:]
        if profile.endswith('tracking'):
            template=template.replace('物理站立跟随','TWIST2 全身物理跟随（实验）')
            template=template.replace('让 G1 站住，再跟随你的动作。','实验：用全身控制策略跟随姿态。')
            template=template.replace('双臂、手腕和腰部跟随你的姿态；双腿优先保持双脚接地和平衡，并随屈膝做有限下蹲。',
                '29 个关节的相对动作作为全身参考；双腿也参与跟随，不经过旧的抬脚启动门槛。')
            template=template.replace('导出 unitree_g1_physics_mapping.json','导出 unitree_g1_tracking_mapping.json')
            start=template.index('<p>物理模式启用重力')
            end=template.index('</p>',start)+4
            template=template[:start]+'<p>此入口使用 TWIST2 配套模型和预训练全身控制策略。标定姿态作为人体中立参考；方向和范围决定相对动作，偏置不会单独改变机器人待机姿态。双腿按协调比例限制到约 30°偏移，上身约 60°。当前原地跟随姿态，不跟踪你的全局行走路线；平衡时机器人可能移动。预览是映射参考，实际由策略、接触与力矩约束决定。<br>双击 start_unitree_g1_tracking.cmd 后走入镜头，自动留出 8 秒就位并等待站稳标定。先轻微转移重心，再慢慢交替提膝和摆臂；空格暂停，C 重新标定，N 后 C 重选，R 重置，Q 退出。此为实验入口，25 秒真人相机动态检查出现跌倒，尚不能作为日常跟随；默认迈步入口为 start_unitree_g1.cmd。</p>'+template[end:]
        joints_file='unitree_g1_tracking_joints.json' if profile.endswith('tracking') else 'unitree_g1_joints.json'
        values = {"__CHANNELS__": CHANNEL_LABELS,
                  "__JOINTS__": json.loads((ROOT / joints_file).read_text(encoding="utf-8")),
                  "__CONFIG__": json.loads((ROOT / (profile + "_mapping.json")).read_text(encoding="utf-8"))}
        for token, value in values.items():
            template = template.replace(token, json.dumps(value, ensure_ascii=False))
        (ROOT / (profile + "_mapping_editor.html")).write_text(template, encoding="utf-8")
        print(profile + " editor generated")


if __name__ == "__main__":
    main()
