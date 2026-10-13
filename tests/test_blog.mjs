import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {dot,unit,residual,cosine,projectionCase,depthCase,fusedPair,sharedReferenceCosine,marginChange} from '../docs/math.mjs';
let checks=0;
const close=(a,b,eps=1e-10)=>{assert.ok(Math.abs(a-b)<eps,`${a} != ${b}`);checks++;};
const base=projectionCase('original'),good=projectionCase('x'),bad=projectionCase('z');
assert.ok(base.relevant<base.competitor);checks++;
close(base.relevant,4/Math.sqrt(20));close(base.competitor,.9);
close(good.relevant,1);close(good.competitor,0);
assert.ok(bad.relevant<bad.competitor);checks++;
assert.ok(bad.relevant-bad.competitor<base.relevant-base.competitor);checks++;
close(depthCase(0).relevant,9/Math.sqrt(126));
close(depthCase(1).relevant,1);close(depthCase(1).competitor,1/Math.sqrt(5));
close(depthCase(2).relevant,1);close(depthCase(2).competitor,1);
const query=JSON.parse(readFileSync(new URL('../docs/data/experiment.json',import.meta.url))).query;
const measured=fusedPair(query,.25),zero=fusedPair(query,0);
close(measured.relevant,query.documents[0].near_score,1e-5);
close(measured.competitor,query.documents[1].near_score,1e-5);
assert.ok(measured.margin>0&&zero.margin<0);checks++;
close(fusedPair(query,measured.crossover).margin,0);
assert.ok(fusedPair(query,measured.crossover-.001).margin<0);checks++;
assert.ok(fusedPair(query,measured.crossover+.001).margin>0);checks++;
// Check the displayed derivation against direct vector arithmetic with
// varied signs and orientations, plus projection idempotence.
for(let i=1;i<=30;i++){
 const q=unit([Math.sin(i),Math.cos(i*.7),.4]),d=unit([.3,Math.sin(i*.9),Math.cos(i)]),c=unit([.2,.4,1]);
 const qr=residual(q,c),dr=residual(d,c);
 close(sharedReferenceCosine(dot(q,d),dot(q,c),dot(d,c)),cosine(qr,dr));
 close(dot(qr,c),0);
 residual(qr,c).forEach((v,j)=>close(v,qr[j]));
 const other=unit([.5,Math.sin(i*.3),.2]),a=dot(q,c),b=dot(d,other);
 const expanded=dot(q,d)-b*dot(q,other)-a*dot(d,c)+a*b*dot(c,other);
 close(expanded,dot(residual(q,c),residual(d,other)));
}
close(sharedReferenceCosine(.98,Math.sqrt(.9),Math.sqrt(.9)),.8);
close(sharedReferenceCosine(.92,Math.sqrt(.9),Math.sqrt(.9)),.2);
for(const next of [-.7,.2,.4,.9]){
 const old=[.3,.5],alpha=.25,mean=x=>x.reduce((a,b)=>a+b)/x.length;
 close(marginChange(old,next,alpha),alpha*(mean([...old,next])-mean(old)));
}
assert.equal(cosine([0,0,0],[1,0,0]),null);checks++;
console.log(`PASS ${checks}: mathematical identities, helpful/harmful projections, depth counterexample, and measured pair crossover.`);
