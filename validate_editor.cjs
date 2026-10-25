// Offline checks of the editor's actual export/preview logic. This lightweight
// DOM model does not substitute for browser layout or file-dialog validation.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
class Element {
  constructor(tag) { this.tagName = tag; this.attributes = {}; this.children = []; this.textContent = ''; this.style = {}; this.listeners = {}; this._value = undefined; }
  setAttribute(key, value) { this.attributes[key] = String(value); if (key === 'value') this._value = String(value); }
  get value() { return this._value === undefined ? (this.tagName === 'select' && this.children.length ? this.children[0].attributes.value : '') : this._value; }
  set value(value) { this._value = String(value); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  click() { if (this.onclick) this.onclick(); }
  matches(selector) {
    if (selector.startsWith('.')) return (this.attributes.class || '').split(' ').includes(selector.slice(1));
    const match = selector.match(/^\[([^=\]]+)(?:="([^"]*)")?\]$/);
    return match ? Object.hasOwn(this.attributes, match[1]) && (match[2] === undefined || this.attributes[match[1]] === match[2]) : false;
  }
  querySelectorAll(selector) { return this.children.flatMap(child => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]); }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
async function main() {
  const filename = process.argv[2] || 'mapping_editor.html';
  const agibot = path.basename(filename) === 'agibot_x1_mapping_editor.html';
  const expectedRules = agibot ? 4 : 3;
  const expectedJoint = agibot ? 'left_elbow_pitch_joint' : 'elbow';
  const expectedElbow = agibot ? 'left_elbow' : 'right_elbow';
  const before = agibot ? 61.1 : 72;
  const after = agibot ? 33.9 : 48;
  const maximum = agibot ? 95 : 120;
  const roots = Object.fromEntries(['add', 'export', 'import', 'manifest', 'rules', 'message', 'confidence', 'smoothing'].map(id => [id, new Element('div')]));
  roots.confidence.value = 0.6; roots.smoothing.value = 0.12;
  const document = {
    createElement: tag => new Element(tag), getElementById: id => roots[id],
    querySelectorAll: selector => Object.values(roots).flatMap(root => root.querySelectorAll(selector)),
  };
  let exported;
  const context = { document, Blob, URL: { createObjectURL: blob => { exported = blob; return 'test:export'; }, revokeObjectURL() {} }, setTimeout: fn => fn() };
  const source = fs.readFileSync(path.join(__dirname, filename), 'utf8').split('<script>')[1].split('</script>')[0];
  vm.runInNewContext(source, context);
  assert.equal(roots.rules.children.length, expectedRules);
  const elbow = roots.rules.children[2];
  assert.equal(elbow.querySelector('.result').textContent, `人体 90° → 机器人 ${before.toFixed(1)}°`);
  const reverse = elbow.querySelectorAll('.ranges')[0]?.children.find(n => n.tagName === 'button');
  // The DOM helper only needs the existing class query used by the editor.
  assert.ok(reverse);
  reverse.click();
  assert.equal(elbow.querySelector('.result').textContent, `人体 90° → 机器人 ${after.toFixed(1)}°`);
  roots.export.click();
  const config = JSON.parse(await exported.text());
  assert.deepEqual(config.rules[2].robot_range_deg, [0, maximum]);
  assert.equal(config.rules[2].robot_joint, expectedJoint);
  assert.equal(config.rules[2].source.landmarks[1], expectedElbow);
  if (agibot) assert.deepEqual(config.viewer.lookat, [0, 0, 0.75]);
  roots.add.click();
  roots.rules.children.at(-1).querySelector('[data-key="joint"]').value = expectedJoint;
  roots.export.click();
  assert.match(roots.message.textContent, /机器人关节重复/);
  await roots.import.onchange({ target: { files: [{ text: async () => JSON.stringify(config) }] } });
  assert.equal(roots.rules.children.length, expectedRules);
  roots.export.click();
  assert.deepEqual(JSON.parse(await exported.text()), config);
  const report = {
    editor: filename, script_syntax: 'passed', initial_rules: expectedRules,
    elbow_preview_deg: [before, after], reverse_and_export: 'passed',
    duplicate_robot_joint_rejection: 'passed', import_export_roundtrip: 'passed',
    browser_layout_and_native_file_dialogs: 'not covered by this offline check',
  };
  const output = path.join(__dirname, 'artifacts');
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, agibot ? 'agibot_x1_editor_validation.json' : 'editor_validation.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
