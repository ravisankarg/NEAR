import assert from 'node:assert/strict';
import fs from 'node:fs';
import {mixturePair,pairCrossover} from '../docs/one-hop-math.mjs';
const data=JSON.parse(fs.readFileSync(new URL('../docs/data/one-hop.json',import.meta.url)));
const q=data.examples.find(q=>q.query_id==='834');
const r=q.documents.find(d=>d.id==='5483793'),c=q.documents.find(d=>d.id==='43048059');
let count=0;
function check(condition){assert.ok(condition);count++;}
const at0=mixturePair(r,c,0),atMeasured=mixturePair(r,c,q.settings.weight),cross=pairCrossover(r,c);
check(at0.margin>0);check(atMeasured.margin<0);check(cross>0&&cross<.01);
check(Math.abs(mixturePair(r,c,cross).margin)<1e-14);
check(Math.abs(atMeasured.relevant-r.mixed_score)<1e-14);
check(Math.abs(atMeasured.competitor-c.mixed_score)<1e-14);
check(r.indegree===0&&r.neighbor_mass===0);
check(r.baseline_rank===9&&r.diffusion_rank===11);
check(c.baseline_rank===11&&c.diffusion_rank===9);
for(let i=0;i<=100;i++){
  const weight=i/100,score=mixturePair(r,c,weight);
  const expected=(1-weight)*(r.direct_mass-c.direct_mass)+weight*(r.neighbor_mass-c.neighbor_mass);
  check(Math.abs(score.margin-expected)<1e-14);
}
check(pairCrossover({direct_mass:.2,neighbor_mass:.2},{direct_mass:.1,neighbor_mass:.1})===null);
assert.throws(()=>mixturePair(r,c,-1),RangeError);count++;
check(data.selection.settings.family==='baseline');
for(const family of ['diffusion','degree_diffusion','rocchio','graph_feedback']){
  check(data.methods[family].comparison['recall@10'].improved===0);
  check(data.methods[family].comparison['recall@10'].worsened===1);
}
console.log(`PASS ${count} one-hop blog arithmetic and evidence checks`);
