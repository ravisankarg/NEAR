import {mixturePair} from './one-hop-math.mjs';
const $=id=>document.getElementById(id);
async function load(){
  try{
    const response=await fetch(new URL('data/one-hop.json',import.meta.url));
    if(!response.ok)throw Error('Experiment data unavailable');
    const data=await response.json(),query=data.examples.find(q=>q.query_id==='834');
    const relevant=query.documents.find(d=>d.id==='5483793'),competitor=query.documents.find(d=>d.id==='43048059');
    function update(){
      const weight=Number($('vote-weight').value),r=mixturePair(relevant,competitor,weight);
      $('vote-value').textContent=weight.toFixed(3);
      $('vote-relevant').textContent=r.relevant.toFixed(6);$('vote-competitor').textContent=r.competitor.toFixed(6);
      $('vote-status').textContent=Math.abs(r.margin)<1e-12?'The two documents tie.':`At λ = ${weight.toFixed(3)}, ${r.margin>0?'the relevant document':'the competitor'} leads by ${Math.abs(r.margin).toFixed(6)}.`;
    }
    $('vote-weight').addEventListener('input',update);
    $('restore-weight').addEventListener('click',()=>{$('vote-weight').value=query.settings.weight;update();});
    update();$('vote-lab').dataset.ready='true';
  }catch(error){$('vote-weight').disabled=true;$('restore-weight').disabled=true;$('vote-status').textContent='Interactive data unavailable; the saved measured scores remain in the table.';console.error(error);}
}
load();
const links=[...document.querySelectorAll('.contents nav a')];
const observer=new IntersectionObserver(entries=>{
  const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);
  if(visible.length)links.forEach(a=>a.setAttribute('aria-current',String(a.hash==='#'+visible[0].target.id)));
},{rootMargin:'-5% 0px -60% 0px',threshold:0});
document.querySelectorAll('.chapter').forEach(s=>observer.observe(s));
