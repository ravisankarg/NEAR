export function mixturePair(relevant,competitor,weight){
  if(!Number.isFinite(weight)||weight<0||weight>1)throw new RangeError('Weight must be in [0,1]');
  const score=d=>(1-weight)*d.direct_mass+weight*d.neighbor_mass;
  const r=score(relevant),c=score(competitor);
  return {relevant:r,competitor:c,margin:r-c};
}
export function pairCrossover(relevant,competitor){
  const gap=relevant.direct_mass-competitor.direct_mass;
  const slope=(relevant.neighbor_mass-competitor.neighbor_mass)-gap;
  if(Math.abs(slope)<1e-15)return null;
  const weight=-gap/slope;
  return weight>=0&&weight<=1?weight:null;
}
