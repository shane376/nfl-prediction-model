import hashlib
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import numpy as np
import pytest
from src.provisional import forecast
from src.presentation import kickoff_label, prediction_label, data_status, default_week
from src.season import score_records

CURRENT = pd.Timestamp('2026-11-05T20:00:00Z')

@pytest.fixture
def future(games):
    game = games.iloc[-1].copy()
    game['game_id'] = 'future'
    game['kickoff'] = pd.Timestamp('2026-12-01T18:00:00Z')
    game['prediction_lock_at'] = game.kickoff
    game['status'] = 'scheduled'
    game['home_score'] = game['away_score'] = float('nan')
    game['gameday'] = '2026-12-01'
    return game

@pytest.fixture
def bundle():
    model = Mock()
    model.predict_proba.return_value = np.array([[.3, .7]])
    return {'model': model, 'metadata': {'training_cutoff': '2026-02-01T00:00:00Z',
            'features': ['win_pct_diff', 'point_diff_diff'], 'version': 'reviewed'}}

def run(games, future, bundle, **source):
    return forecast(games, future, {'fetched_at': CURRENT.isoformat(), **source}, bundle, CURRENT)

def test_provisional_uses_observed_inputs_and_never_scores(games, future, bundle):
    row = run(games, future, bundle)
    assert row['kind'] == 'provisional'
    assert row['generated_at'] == CURRENT.isoformat()
    assert row['home_probability'] == .7
    assert row['target'] is None
    assert score_records([{**row, 'target': 1, 'result': 'correct'}])['games'] == 0
    changed = games.copy()
    changed.loc[changed.season.eq(2026), 'available_at'] = CURRENT + pd.Timedelta(days=1)
    run(changed, future, bundle)
    values = bundle['model'].predict_proba.call_args.args[0]
    assert values.iloc[0].eq(0).all()

def test_tbd_requires_future_date(games, future, bundle):
    future['kickoff'] = future['prediction_lock_at'] = pd.NaT
    future['status'] = 'kickoff_unavailable'
    assert run(games, future, bundle)['kind'] == 'provisional'
    future['gameday'] = None
    assert run(games, future, bundle)['kind'] == 'unavailable'
    future['gameday'] = '2026-11-05'
    assert run(games, future, bundle)['kind'] == 'unavailable'

@pytest.mark.parametrize('change', ['stale', 'future_source', 'no_model', 'closed_lock', 'completed', 'future_model', 'invalid_probability'])
def test_provisional_fails_closed(games, future, bundle, change):
    source = {}
    if change == 'stale': source['stale'] = True
    if change == 'future_source': source['fetched_at'] = '2026-12-01T00:00:00Z'
    if change == 'no_model': bundle = None
    if change == 'closed_lock': future['prediction_lock_at'] = CURRENT
    if change == 'completed': future['status'] = 'completed'
    if change == 'future_model': bundle['metadata']['training_cutoff'] = '2027-01-01T00:00:00Z'
    if change == 'invalid_probability': bundle['model'].predict_proba.return_value = np.array([[0, float('nan')]])
    assert run(games, future, bundle, **source)['kind'] == 'unavailable'

def test_presentation(games):
    assert kickoff_label(pd.NaT) == 'Kickoff TBD'
    assert 'EST' in kickoff_label(CURRENT)
    assert prediction_label({'kind': 'live'}) == 'Recorded pregame pick'
    assert prediction_label({'kind': 'live', 'integrity_error': 'late'}) == 'Prediction unavailable'
    assert data_status({}, games, CURRENT)['stale']
    assert default_week(games.iloc[:0], CURRENT) == 1

def test_dashboard_offline_read_only_all_views(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src import data, modeling, predictions, retrospective
    def forbidden(*args, **kwargs):
        raise AssertionError('Network, training or persistence invoked during rendering')
    for module, name in [(data, 'urlopen'), (data, 'write_json'), (modeling, 'train'),
                         (retrospective, 'fit_frozen'), (predictions, 'save_record')]:
        monkeypatch.setattr(module, name, forbidden)
    root = Path(__file__).resolve().parents[1]
    def hashes():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'artifacts').rglob('*') if p.is_file()}
    before = hashes()
    app = AppTest.from_file(str(root/'dashboard/app.py'), default_timeout=30).run()
    assert not app.exception
    app.selectbox(key='week').select(18).run()
    assert not app.exception
    for view in ['Team Explorer', 'Season performance', 'Prediction history', 'Methodology']:
        app.radio[0].set_value(view).run()
        assert not app.exception
    assert before == hashes()
