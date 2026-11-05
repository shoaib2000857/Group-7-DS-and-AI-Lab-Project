"""Protocol/integration tests; fake outputs here are never benchmark findings."""
from pathlib import Path
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import benchmark_runtime as tw


class BenchmarkTests(unittest.TestCase):
    def test_bootstrap_avoids_stdlib_ensurepip_route_and_logs_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(tw,'supervise') as runner:
                tw.create_worker_environment(root/'environments/hf',root/'run','hf')
            commands=[call.args[0] for call in runner.call_args_list]
            self.assertTrue(any('virtualenv==20.35.3' in cmd for cmd in commands))
            create=next(cmd for cmd in commands if 'virtualenv' in cmd)
            self.assertIn('--system-site-packages',create)
            self.assertIn('--no-download',create)
            self.assertFalse(any(cmd[1:3]==['-m','venv'] for cmd in commands))
            self.assertTrue(all(call.args[1].parent.name=='logs' for call in runner.call_args_list))

    def test_shared_bootstrap_failure_stops_before_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            with patch.object(tw,'environment',return_value={'gpu':{'name':'test GPU','total_gb':24,'bf16_supported':True}}), \
                 patch.object(tw,'create_worker_environment',side_effect=RuntimeError('test bootstrap failure')), \
                 patch.object(tw,'archive_results',return_value=run/'partial.zip') as exporter:
                with self.assertRaisesRegex(RuntimeError,'before benchmark data/model inference'):
                    tw.preflight_worker_setup(tw.default_config(),run)
            self.assertTrue(exporter.called)
            self.assertEqual(json.loads((run/'metadata/bootstrap_preflight.json').read_text())['status'],'failed_before_data_or_models')

    def test_reference_fields_are_not_in_inference_payload(self):
        case={'claim_text':'One building appeared.','claim':{'object':'building'},'target':'SECRET',
              'witness':{'summary':'An uncertain building change.','target':'SECRET','source':'witness'},
              'mask':{'visible_change':True,'gold':'SECRET','mask_path':'SECRET'}}
        payload=json.dumps(tw.inference_state(case))
        self.assertNotIn('SECRET',payload)
        self.assertIn('uncertain',payload)

    def test_empty_acceptance_is_undefined_not_perfect(self):
        rows=[]
        for i,target in enumerate(tw.OUTCOMES):
            rows.append({'target':target,'status':'ok','outcome':'defer','probabilities':None,
                         'latency_s':None,'peak_gpu_gb':None,'scene_id':str(i)})
        result=tw.metric_row(pd.DataFrame(rows))
        self.assertIsNone(result['accepted_precision_proxy'])
        self.assertEqual(result['decision_coverage'],0)
        self.assertEqual(result['true_claim_acceptance'],0)
        self.assertEqual(result['deferral_rate'],1)

    def test_multiclass_brier_and_failure_denominators(self):
        frame=pd.DataFrame([
            {'target':'accept','status':'ok','outcome':'accept','probabilities':{'accept':.8,'reject':.1,'defer':.1},'latency_s':1.,'peak_gpu_gb':2.,'scene_id':'a'},
            {'target':'accept','status':'not_completed','outcome':None,'probabilities':None,'latency_s':None,'peak_gpu_gb':None,'scene_id':'b'}])
        result=tw.metric_row(frame)
        self.assertAlmostEqual(result['brier_multiclass_0_to_2'],.06)
        self.assertEqual(result['completion_rate'],.5)
        self.assertEqual(result['true_claim_acceptance'],.5)
        self.assertEqual(result['decision_coverage'],.5)

    def test_native_probabilities_are_validated(self):
        with self.assertRaises(ValueError):
            tw._checked_probabilities({'accept':1.0})
        with self.assertRaises(ValueError):
            tw._checked_probabilities({'accept':.9,'reject':.9,'defer':.9})
        self.assertEqual(tw._parse_label('C',['defer','reject','accept']),'accept')
        with self.assertRaises(ValueError):
            tw._parse_label('A because I agree',tw.OUTCOMES)

    def test_accept_gate_does_not_manufacture_rule_probabilities(self):
        self.assertEqual(tw.policy_outcome({'status':'ok','outcome':'accept','probabilities':None},.8),'accept')
        self.assertEqual(tw.policy_outcome({'status':'ok','outcome':'accept','probabilities':{'accept':.7,'reject':.2,'defer':.1}},.8),'defer')
        self.assertEqual(tw.policy_outcome({'status':'ok','outcome':'reject','probabilities':{'accept':.2,'reject':.7,'defer':.1}},.8),'reject')

    def test_fixtures_keep_pair_groups_disjoint(self):
        cases,refs=tw.synthetic_fixture()
        phase={}
        for c in cases:
            self.assertEqual(phase.setdefault(c['pair_id'],c['phase']),c['phase'])
        targets={r['case_id']:r['target'] for r in refs}
        for case in cases:
            self.assertEqual(tw.rule_decision(case,'R02'),targets[case['case_id']])
        self.assertEqual(len(set(c['case_id'] for c in cases)),len(cases))

    def test_zip_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('../outside.txt','escape')
            with self.assertRaises(ValueError):
                tw.safe_unpack(path,Path(folder)/'inside')

    def test_archive_checksums_remain_valid_after_repeat_export(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'run';run.mkdir()
            tw.write_json(run/'config.json',{'test':True})
            (run/'large_weights.bin').write_bytes(b'not-an-output')
            for _ in range(2):
                output=tw.archive_results(run,False)
                with zipfile.ZipFile(output) as archive:
                    self.assertNotIn('large_weights.bin',archive.namelist())
                    manifest=json.loads(archive.read('EXPORT_MANIFEST.json'))
                    for item in manifest['files']:
                        self.assertEqual(item['sha256'],tw.digest(archive.read(item['path'])))

    def test_missing_learned_outputs_remain_visible(self):
        cases,refs=tw.synthetic_fixture()
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            (run/'predictions').mkdir()
            config=tw.default_config();config['MODEL_IDS']=['J05']
            frame=tw.prediction_frame(cases,refs,config,run)
            learned=frame[(frame['model_id']=='J05') & (frame['track']=='common_evidence')]
            self.assertEqual(len(learned),len(cases))
            self.assertTrue((learned['status']=='not_completed').all())

    def test_resume_signature_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            config=tw.default_config()
            run,config=tw.start_run(config,Path(folder))
            config['RESUME_RUN']=str(run)
            resumed,_=tw.start_run(config,Path(folder))
            self.assertEqual(run,resumed)
            config['ACCEPT_THRESHOLD']=.5
            with self.assertRaises(ValueError):
                tw.start_run(config,Path(folder))

    def test_partial_prediction_can_be_recovered_before_append(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'predictions/model.jsonl'
            path.parent.mkdir()
            path.write_text('{"status":"ok","id":1}\n{"status":',encoding='utf-8')
            self.assertEqual(len(tw.read_jsonl(path)),1)
            tw.append_jsonl(path,{'status':'ok','id':2})
            self.assertEqual([r['id'] for r in tw.read_jsonl(path)],[1,2])
            self.assertTrue(path.with_suffix('.interrupted_fragment.txt').exists())

    def test_raw_and_thresholded_metrics_remain_separate(self):
        # Manufactured outputs are a parser/evaluator test, never a model run.
        cases,refs=tw.synthetic_fixture()
        cases=cases[:2]
        for case in cases:
            case.update(phase='val',benchmark_group='evaluator_test_fixture')
        refs=refs[:2]
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            for name in ['inputs','predictions','references','metadata','tables','logs']:
                (run/name).mkdir()
            config=tw.default_config();config['MODEL_IDS']=['J05']
            tw.run_rules(cases,run)
            for case in cases:
                tw.append_jsonl(run/'predictions/J05_judge.jsonl',{
                    'model_id':'J05','track':'common_evidence','case_id':case['case_id'],'status':'ok','outcome':'accept',
                    'probabilities':{'accept':.7,'reject':.2,'defer':.1},'latency_s':.1,'peak_gpu_gb':1.0})
            frame,summary,register=tw.evaluate(cases,refs,config,run)
            learned=frame[(frame['model_id']=='J05')&(frame['track']=='common_evidence')]
            self.assertTrue((learned['raw_outcome']=='accept').all())
            self.assertTrue((learned['outcome']=='defer').all())
            raw=pd.read_csv(run/'tables/raw_choice_metrics.csv')
            row=raw[(raw['model_id']=='J05')&(raw['track']=='common_evidence')].iloc[0]
            self.assertEqual(row['true_claim_acceptance'],1.)
            self.assertEqual(tw.metric_row(learned)['true_claim_acceptance'],0.)


if __name__=='__main__':
    unittest.main()
