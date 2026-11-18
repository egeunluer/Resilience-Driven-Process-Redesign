"""
Artifact paths per event log. The scripts (training, baselines, calibration, shadow
price) pick the log with the environment variable RIMS_LOG (default bpic2017):
the stage model, the disruption config and the resilience calibration of that log
in Module 1/artifacts. The method code is the same for every log.
"""
import os

ART = '../../Module 1/artifacts/'
LOGS = {
    'bpic2017': {'model': 'bpic2017_stage_model.json', 'disruptions': 'bpic2017_disruptions.json',
                 'calibration': 'bpic2017_pool3_resilience_calibration.json'},
    'bpi2012': {'model': 'bpi2012_stage_model.json', 'disruptions': 'bpi2012_disruptions.json',
                'calibration': 'bpi2012_pool3_resilience_calibration.json'},
    # Phase 2 only so far: disruption config and calibration are written in later phases
    'consulta': {'model': 'consulta_stage_model.json', 'disruptions': 'consulta_disruptions.json',
                 'calibration': 'consulta_resilience_calibration.json'},
}


def current_log():
    log = os.environ.get('RIMS_LOG', 'bpic2017')
    if log not in LOGS:
        raise ValueError(f'RIMS_LOG={log!r}: known logs are {sorted(LOGS)}')
    return log


def paths(log=None):
    """(stage model, disruption config, calibration) of the log, default RIMS_LOG."""
    files = LOGS[log or current_log()]
    return ART + files['model'], ART + files['disruptions'], ART + files['calibration']
