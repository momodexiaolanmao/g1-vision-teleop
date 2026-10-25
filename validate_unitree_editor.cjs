// Exercise actual editor source with a minimal DOM. Native dialogs/layout are
// checked separately; this validates mapping preservation and transformations.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
class Element{
 constructor(tag){this.tagName=tag;this.attributes={};this.children=[];this.textContent='';this.style={};this.listeners={};this._value=undefined}
 setAttribute(k,v){this.attributes[k]=String(v);if(k==='value')this._value=String(v)}
 get value(){return this._value===undefined?(this.tagName==='select'&&this.children.length?this.children[0].attributes.value:''):this._value}
 set value(v){this._value=String(v)}
 append(...items){this.children.push(...items)}
 replaceChildren(...items){this.children=items}
 addEventListener(name,fn){this.listeners[name]=fn}
 click(){this.onclick?.()}
 matches(s){if(s.startsWith('.'))return(this.attributes.class||'').split(' ').includes(s.slice(1));const m=s.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);return m?Object.hasOwn(this.attributes,m[1])&&(m[2]===undefined||this.attributes[m[1]]===m[2]):false}
 querySelectorAll(s){return this.children.flatMap(c=>[...(c.matches(s)?[c]:[]),...c.querySelectorAll(s)])}
 querySelector(s){return this.querySelectorAll(s)[0]||null}
}
async function main(){
 const profile=process.argv[2]||'unitree_g1';
 const roots=Object.fromEntries(['add','export','import','rules','message','confidence','smoothing','depth','distance','group','count'].map(k=>[k,new Element('div')]));roots.group.value='all';
 const document={createElement:t=>new Element(t),getElementById:id=>roots[id],querySelectorAll:s=>Object.values(roots).flatMap(r=>r.querySelectorAll(s))};let exported;
 const context={document,Blob,URL:{createObjectURL:b=>{exported=b;return'test:export'},revokeObjectURL(){}},setTimeout:fn=>fn()};
 const html=fs.readFileSync(path.join(__dirname,profile+'_mapping_editor.html'),'utf8');vm.runInNewContext(html.split('<script>')[1].split('</script>')[0],context);
 const original=JSON.parse(fs.readFileSync(path.join(__dirname,profile+'_mapping.json'),'utf8'));
 assert.equal(roots.rules.children.length,29);roots.export.click();assert.deepEqual(JSON.parse(await exported.text()),original);
 for(const[group,count]of[['legs',12],['arms',14],['waist',3]]){roots.group.value=group;roots.group.onchange();assert.equal(roots.rules.children.filter(c=>c.style.display!=='none').length,count)}
 roots.group.value='all';roots.group.onchange();const elbow=roots.rules.children[18];assert.match(elbow.querySelector('.result').textContent,/G1 45.0°/);
 elbow.querySelector('.ranges').children.find(n=>n.tagName==='button').click();assert.match(elbow.querySelector('.result').textContent,/G1 -15.0°/);
 elbow.querySelector('[data-key="channel"]').value='right_knee_flex';roots.export.click();const changed=JSON.parse(await exported.text());assert.equal(changed.rules[18].source.channel,'right_knee_flex');assert.deepEqual(changed.rules[18].robot_range_deg,[-60,90]);assert.equal(changed.control_mode,original.control_mode);assert.deepEqual(changed.root_tracking,original.root_tracking);assert.deepEqual(changed.balance,original.balance);
 roots.rules.children[19].querySelector('[data-key="joint"]').value=changed.rules[18].robot_joint;roots.export.click();assert.match(roots.message.textContent,/机器人关节重复/);
 await roots.import.onchange({target:{files:[{text:async()=>JSON.stringify(changed)}]}});roots.export.click();assert.deepEqual(JSON.parse(await exported.text()),changed);
 roots.rules.children[0].querySelector('.top').children.find(n=>n.tagName==='button').click();assert.equal(roots.rules.children.length,28);roots.add.click();assert.equal(roots.rules.children.length,29);roots.export.click();assert.equal(JSON.parse(await exported.text()).rules.length,29);
 const report={editor:profile+'_mapping_editor.html',initial_rules:29,groups:{legs:12,arms:14,waist:3},reverse_preview_deg:[45,-15],human_channel_remap:'passed',duplicate_joint_rejection:'passed',import_export_roundtrip:'passed',mode_and_controller_settings_preservation:'passed',remove_and_add_rule:'passed',native_dialogs:'not covered by offline check'};
 fs.writeFileSync(path.join(__dirname,'artifacts/'+profile+'_editor_validation.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}
main().catch(e=>{console.error(e);process.exitCode=1});
