import {projectionCase,depthCase,fusedPair} from './math.mjs';
const $=id=>document.getElementById(id);
const signed=(x,d=4)=>(x<0?'−':'+')+Math.abs(x).toFixed(d);
function activate(buttons,current){buttons.forEach(b=>b.setAttribute('aria-pressed',String(b===current)));}
const projectionButtons=[...document.querySelectorAll('[data-projection]')];
projectionButtons.forEach(button=>button.addEventListener('click',()=>{
  activate(projectionButtons,button);const mode=button.dataset.projection,r=projectionCase(mode);
  $('toy-relevant').textContent=r.relevant.toFixed(4);$('toy-competitor').textContent=r.competitor.toFixed(4);
  $('toy-relevant-bar').style.width=r.relevant*100+'%';$('toy-competitor-bar').style.width=r.competitor*100+'%';
  $('toy-status').textContent={original:'The original embeddings slightly favor the competitor.',x:'The relevant residual aligns perfectly with the query. The competitor is perpendicular.',z:'Removing the competitor’s mismatch makes the wrong preference stronger.'}[mode];
  $('toy-vectors').textContent=r.vectors.map((v,i)=>['q','R','C'][i]+' = ('+v.join(', ')+')').join(' · ');
}));
const depthButtons=[...document.querySelectorAll('[data-depth]')];
depthButtons.forEach(button=>button.addEventListener('click',()=>{
  activate(depthButtons,button);const depth=Number(button.dataset.depth),r=depthCase(depth);
  $('depth-relevant').textContent=r.relevant.toFixed(4);$('depth-competitor').textContent=r.competitor.toFixed(4);
  $('depth-margin').textContent=Math.abs(r.relevant-r.competitor)<1e-12?'0.0000':signed(r.relevant-r.competitor);
  $('depth-status').textContent=['The competitor wins on the original embeddings.','The first removal fixes the ordering: relevant margin +0.5528.','Both cosines are 1. This new signal cannot distinguish the documents.'][depth];
}));
async function loadMeasured(){
  try{
    const response=await fetch(new URL('data/experiment.json',import.meta.url));if(!response.ok)throw Error('Data unavailable');
    const data=await response.json(),query=data.query;
    function update(){
      const alpha=Number($('alpha').value),r=fusedPair(query,alpha);$('alpha-value').textContent=alpha.toFixed(3);
      $('relevant-score').textContent=r.relevant.toFixed(4);$('competitor-score').textContent=r.competitor.toFixed(4);
      // Track center represents zero, with symmetric display bounds.
      $('margin-indicator').style.left=(50+Math.max(-1,Math.min(1,r.margin/.5))*50)+'%';
      $('pair-status').textContent=Math.abs(r.margin)<1e-8?'The documents tie under this scoring rule.':`At α = ${alpha.toFixed(3)}, ${r.margin>0?'the relevant document':'the competitor'} leads by ${Math.abs(r.margin).toFixed(4)}.`;
    }
    $('alpha').addEventListener('input',update);$('reset-alpha').addEventListener('click',()=>{$('alpha').value=query.alpha;update();});update();
    $('query-lab').dataset.ready='true';
  }catch(error){$('alpha').disabled=true;$('reset-alpha').disabled=true;$('pair-status').textContent='Live controls are unavailable. The saved measured comparison remains above.';console.error(error);}
}
loadMeasured();
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{
  try{await navigator.clipboard.writeText($(button.dataset.copy).innerText);button.textContent='Copied';setTimeout(()=>button.textContent='Copy',1800);}
  catch{$('toast').textContent='Select the code to copy it.';$('toast').classList.add('visible');setTimeout(()=>$('toast').classList.remove('visible'),2500);}
}));
const links=[...document.querySelectorAll('.contents nav a')];
const observer=new IntersectionObserver(entries=>{
  const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);
  if(visible.length)links.forEach(a=>a.setAttribute('aria-current',String(a.hash==='#'+visible[0].target.id)));
},{rootMargin:'-5% 0px -60% 0px',threshold:0});
document.querySelectorAll('.chapter').forEach(s=>observer.observe(s));
