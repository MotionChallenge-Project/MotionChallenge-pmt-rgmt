import ast
import hashlib
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace as NS
import numpy as np

def source_definitions(path, names, scope):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('missing native definitions')
    for n in nodes:
        # Registry and JIT decoration do not change the evaluated formulas.
        n.decorator_list = []
    prefix = ast.parse('from __future__ import annotations').body
    exec(compile(ast.Module(body=prefix + nodes, type_ignores=[]), str(path), 'exec'), scope)

def validate_control_config(source, root, adapter, pinned):
    """Re-evaluate the pinned author's config, using inert config containers."""
    class Config(NS):
        pass
    Config.InitialStateCfg = Config
    Config.JointDriveCfg = Config
    Config.PDGainsCfg = Config
    scope = {'ASSET_DIR': '', 'ArticulationCfg': Config, 'ImplicitActuatorCfg': Config,
             'sim_utils': NS(**{n:Config for n in ['UrdfFileCfg','UsdFileCfg','RigidBodyPropertiesCfg',
                        'ArticulationRootPropertiesCfg','UrdfConverterCfg',
                        'CollisionPropertiesCfg','PreviewSurfaceCfg']})}
    tree = ast.parse((source/'pmt_tasks/robots/g1.py').read_text())
    tree.body = [n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))
                 and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                          and isinstance(n.value.func,ast.Name) and n.value.func.id=='print')]
    exec(compile(tree, 'native-robot-config', 'exec'),scope)
    robot = scope['G1_CYLINDER_CFG']
    def match(values,name,default=None):
        if not isinstance(values,dict): return values
        hits = [v for pattern,v in values.items() if re.fullmatch(pattern,name)]
        if not hits and default is not None: return default
        if len(hits)!=1: raise ValueError('ambiguous native config: '+name)
        return hits[0]
    defaults=[];scales=[];kps=[];kds=[]
    for name in adapter.names:
        matches=[a for a in robot.actuators.values()
                 if any(re.fullmatch(pattern,name) for pattern in a.joint_names_expr)]
        assert len(matches)==1
        actuator=matches[0]
        defaults.append(match(robot.init_state.joint_pos,name,0.))
        scales.append(match(scope['G1_ACTION_SCALE'],name))
        kps.append(match(actuator.stiffness,name));kds.append(match(actuator.damping,name))
    for actual,expected in [(adapter.default,defaults),
                            (adapter.kp,kps),(adapter.kd,kds)]:
        np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-7)
    xml=root/'motion-challenge/robots/g1-29dof-cpu-v0.xml'
    assert hashlib.sha256(xml.read_bytes()).hexdigest()==pinned['canonical_robot_xml_sha256']
    pelvis=ET.parse(xml).find('.//body[@name="pelvis"]/inertial')
    np.testing.assert_array_equal(adapter.cfg['canonical_root_com_offset_m'],
                                  [float(v) for v in pelvis.attrib['pos'].split()])
    return {'native_pd_default':True,'canonical_pelvis_com':True}
