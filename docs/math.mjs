/** Pure arithmetic for the explanatory labs. No encoder or retrieval runs here. */
export function dot(a,b){return a.reduce((s,x,i)=>s+x*b[i],0);}
export function unit(a){const n=Math.sqrt(dot(a,a));return n>1e-12?a.map(x=>x/n):a.map(()=>0);}
export function cosine(a,b){if(dot(a,a)<1e-24||dot(b,b)<1e-24)return null;return Math.max(-1,Math.min(1,dot(unit(a),unit(b))));}
export function residual(a,c){const direction=unit(c),scale=dot(a,direction);return a.map((x,i)=>x-scale*direction[i]);}
export function projectionCase(mode){
  const vectors=[[3,1,0],[1,1,0],[3,0,1]];
  const c=mode==='x'?[1,0,0]:mode==='z'?[0,0,1]:null;
  const [q,r,n]=c?vectors.map(v=>residual(v,c)):vectors;
  return {vectors:[q,r,n],relevant:cosine(q,r),competitor:cosine(q,n)};
}
export function depthCase(depth){
  let vectors=[[4,2,1],[1,2,1],[4,0,1]];
  for(const c of [[1,0,0],[0,1,0]].slice(0,depth))vectors=vectors.map(v=>residual(v,c));
  const [q,r,n]=vectors;
  return {vectors,relevant:cosine(q,r),competitor:cosine(q,n)};
}
export function fusedPair(query,alpha){
  const [r,c]=query.documents;
  const score=d=>d.baseline_cosine/query.tau_original+alpha*d.residual_cosine/query.tau_normalized_residual;
  const relevant=score(r),competitor=score(c);
  const crossover=(c.baseline_cosine-r.baseline_cosine)/query.tau_original/((r.residual_cosine-c.residual_cosine)/query.tau_normalized_residual);
  return {relevant,competitor,margin:relevant-competitor,crossover};
}
export function sharedReferenceCosine(s,a,b){return(s-a*b)/Math.sqrt((1-a*a)*(1-b*b));}
export function marginChange(previous,next,alpha){const mean=previous.reduce((a,b)=>a+b,0)/previous.length;return alpha/(previous.length+1)*(next-mean);}
