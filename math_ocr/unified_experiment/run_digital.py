"""Single experiment entry point after user explicitly excluded photos."""
import json,hashlib,sys,subprocess
from engine import ROOT
from synthetic import prepare
from training import train,save

def main():
    existing=json.loads((ROOT/'state.json').read_text(encoding='utf-8')) if (ROOT/'state.json').exists() else {}
    if existing.get('state')=='COMPLETE_WITH_LIMITATIONS' and (ROOT/'REPORT.md').exists():
        print('This experiment is complete; existing frozen results are preserved. See REPORT.md.');return
    real=json.loads((ROOT/'prepared_manifest.json').read_text(encoding='utf-8'))
    if not (ROOT/'symbol_parent_audit.json').exists():raise RuntimeError('Run symbol parent audit before locking data.')
    synthetic=prepare()
    signature=hashlib.sha256(json.dumps(real+synthetic,sort_keys=True).encode()).hexdigest()
    save('data_lock.json',{'manifest_hash':signature,'real_counts':{s:sum(r['split']==s for r in real) for s in ['train','val','test']},'synthetic_training_count':len(synthetic),'photos_used':0,'authorization':'User: 직접 촬영 사진은 없이 진행해','selection':'MathWriting validation exact match, then normalized token edit distance'})
    if not (ROOT/'final_model.pt').exists():
        save('state.json',{'state':'TRAINING','complete':False,'photos_used':0,'data_lock_hash':signature,'training_started':True,'test_predictions_generated':False});train(real,synthetic)
    elif not (ROOT/'mathwriting_test.json').exists():
        raise RuntimeError('Frozen checkpoint exists without test record. Restore matching state before continuing; do not silently retrain.')
    if not (ROOT/'REPORT.md').exists():
        from finalize_digital import main as finalize
        finalize()
    if not (ROOT/'ui_checks.json').exists():
        subprocess.run([sys.executable,str(ROOT/'verify_web_api.py'),'--self-host','--port','8768'],check=True)
        subprocess.run(['node',str(ROOT/'verify_ui_logic.cjs')],check=True)
    ui=json.loads((ROOT/'ui_checks.json').read_text(encoding='utf-8'))
    if not ui.get('js_unit_checks'):
        subprocess.run(['node',str(ROOT/'verify_ui_logic.cjs')],check=True);ui=json.loads((ROOT/'ui_checks.json').read_text(encoding='utf-8'))
    primary=json.loads((ROOT/'mathwriting_test.json').read_text(encoding='utf-8'))
    required=['final_model.pt','frozen_selection.json','mathwriting_test.json','diagnostics_crohme.json','paired_fraction_diagnostic.json','latency.json','test_gallery.html','REPORT.md']
    if any(not (ROOT/name).exists() for name in required):raise RuntimeError('Required full-experiment artifact missing')
    save('state.json',{'state':'COMPLETE_WITH_LIMITATIONS','complete':True,'protocol_version':'3.0','training_started':True,'test_predictions_generated':True,'photos_used':0,'primary_test_n':primary['n'],'primary_correct':primary['correct_count'],'primary_exact_match':primary['exact_match'],'performance_target_met':False,'api_and_js_checks_passed':bool(ui['api_status_ready'] and ui['js_unit_checks']['response_display']),'live_browser_clicks_verified':False,'live_browser_limitation':ui['browser_limitation'],'app_url':ui['url']})
    print('Full digital-ink experiment completed with documented accuracy and browser-verification limitations. See REPORT.md.')

if __name__=='__main__':main()
