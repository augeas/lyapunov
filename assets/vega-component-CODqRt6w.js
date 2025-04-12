import{an as e,ao as t,G as n,ap as a,P as r,ac as i,J as o,aq as s,ar as c,as as l,at as u,au as m,av as p,aw as d,ax as f,ay as h,az as g,aA as y,aB as v}from"./index-DtPBW9Ge.js";import{M as k}from"./compile-hxmSbHOR.js";import{a as w}from"./VegaLite-RFaMP3Qr.js";import"./time-DKLM0YDh.js";import"./step-DY8hpDjU.js";import"./linear-BpDp71Jt.js";import"./init-DLRA0X12.js";import"./range-CtcPcB_L.js";import"./ordinal-BcaZTuz9.js";import"./colors-bszWmPJw.js";import"./arc-y9WbYkJu.js";import"./index-BTmvn_00.js";const x={getMarkType(e){const t="string"==typeof e?e:e.type;if("boxplot"===t||"errorband"===t||"errorbar"===t)throw new Error("Not supported");return t},makeClickable(e){const t="string"==typeof e?e:e.type;return t in k?"string"==typeof e?{type:e,cursor:"pointer",tooltip:!0}:{...e,type:t,cursor:"pointer",tooltip:!0}:e},getOpacity:e=>"string"==typeof e?null:"opacity"in e&&"number"==typeof e.opacity?e.opacity:null};const j=new Set(["color","fill","fillOpacity","opacity","shape","size"]);function b(e,t,n,a){const r={and:n.map((e=>({param:e})))};if("opacity"===e){const e=x.getOpacity(a)||1;return{...t,opacity:{condition:{test:r,value:e},value:e/5}}}return t}const S={point:e=>null==e?"select_point":`select_point_${e}`,interval:e=>null==e?"select_interval":`select_interval_${e}`,legendSelection:e=>`legend_selection_${e}`,HIGHLIGHT:"highlight",PAN_ZOOM:"pan_zoom",hasPoint:e=>e.some((e=>e.startsWith("select_point"))),hasInterval:e=>e.some((e=>e.startsWith("select_interval"))),hasLegend:e=>e.some((e=>e.startsWith("legend_selection"))),hasPanZoom:e=>e.some((e=>e.startsWith("pan_zoom")))},_={highlight:()=>({name:S.HIGHLIGHT,select:{type:"point",on:"mouseover"}}),interval:(e,t)=>({name:S.interval(t),select:{type:"interval",encodings:P(e),mark:{fill:"#669EFF",fillOpacity:.07,stroke:"#669EFF",strokeOpacity:.4},on:"[mousedown[!event.metaKey], mouseup] > mousemove[!event.metaKey]",translate:"[mousedown[!event.metaKey], mouseup] > mousemove[!event.metaKey]"}}),point:(e,t)=>({name:S.point(t),select:{type:"point",encodings:P(e),on:"click[!event.metaKey]"}}),legend:e=>({name:S.legendSelection(e),select:{type:"point",fields:[e]},bind:"legend"}),panZoom:()=>({name:S.PAN_ZOOM,bind:"scales",select:{type:"interval",on:"[mousedown[event.metaKey], window:mouseup] > window:mousemove!",translate:"[mousedown[event.metaKey], window:mouseup] > window:mousemove!",zoom:"wheel![event.metaKey]"}})};function P(e){switch(x.getMarkType(e.mark)){case k.image:case k.trail:return;case k.area:case k.arc:return["color"];case k.bar:{const t=function(e){var t,n;if(!e||!("mark"in e))return;const a=null==(t=e.encoding)?void 0:t.x,r=null==(n=e.encoding)?void 0:n.y;if(a&&"type"in a&&"nominal"===a.type)return"vertical";if(r&&"type"in r&&"nominal"===r.type)return"horizontal";if(a&&"aggregate"in a)return"horizontal";if(r&&"aggregate"in r)return"vertical";return}(e);return"horizontal"===t?["y"]:"vertical"===t?["x"]:void 0}case k.circle:case k.geoshape:case k.line:case k.point:case k.rect:case k.rule:case k.square:case k.text:case k.tick:return["x","y"]}}function M(e){if("params"in e&&e.params&&e.params.length>0){return e.params.filter((e=>null!=e&&("select"in e&&void 0!==e.select))).map((e=>e.name))}return"layer"in e?[...new Set(e.layer.flatMap(M))]:[]}function O(e,t){var n,a;let{chartSelection:r=!0,fieldSelection:i=!0}=t;if(!r&&!i)return e;(null==(n=e.params)?void 0:n.some((e=>"legend"===e.bind)))&&(i=!1);if((null==(a=e.params)?void 0:a.some((e=>!e.bind)))&&(r=!1),"vconcat"in e){const t=e.vconcat.map((e=>"mark"in e?z(e):e));return{...e,vconcat:t}}if("hconcat"in e){const t=e.hconcat.map((e=>"mark"in e?z(e):e));return{...e,hconcat:t}}if("layer"in e){const t=e.layer.map(((e,t)=>{if(!("mark"in e))return e;let n=e;return n=N(n,r,t),n=z(n),0===t&&(n=V(n)),n}));return{...e,layer:t}}if(!("mark"in e))return e;let o=e;return o=function(e,t){if(!1===t)return e;let n=function(e){if(!e||!("encoding"in e))return[];const{encoding:t}=e;return t?Object.entries(t).flatMap((e=>{const[t,n]=e;return n&&j.has(t)?"field"in n&&"string"==typeof n.field?[n.field]:"condition"in n&&n.condition&&"object"==typeof n.condition&&"field"in n.condition&&n.condition.field&&"string"==typeof n.condition.field?[n.condition.field]:[]:[]})):[]}(e);Array.isArray(t)&&(n=n.filter((e=>t.includes(e))));const a=n.map((e=>_.legend(e))),r=[...e.params||[],...a];return{...e,params:r}}(o,i),o=N(o,r,void 0),o=z(o),o=V(o),o}function N(e,t,n){if(!1===t)return e;let a;try{a=x.getMarkType(e.mark)}catch{return e}if("geoshape"===a||"text"===a)return e;const r=!0===t?function(e){switch(e){case"text":case"arc":case"area":return["point"];case"bar":default:return["point","interval"];case"line":return}}(a):[t];if(!r)return e;const i=r.map((t=>"interval"===t?_.interval(e,n):_.point(e,n))),o=[...e.params||[],...i];return{...e,params:o}}function V(e){let t;try{t=x.getMarkType(e.mark)}catch{}if("geoshape"===t)return e;const n=e.params||[];return n.some((e=>"scales"===e.bind))?e:{...e,params:[...n,_.panZoom()]}}function z(e){const t="encoding"in e?e.encoding:void 0,n=e.params||[],a=n.map((e=>e.name));if(0===n.length)return e;return"text"===x.getMarkType(e.mark)?e:{...e,mark:x.makeClickable(e.mark),encoding:b("opacity",t||{},a,e.mark)}}const A=i=>{const o=n(11),{value:s,setValue:c,chartSelection:l,fieldSelection:u,spec:m}=i;let p,f;o[0]!==m?(p=async()=>async function(n){if(!n)return n;const a="datasets"in n?{...n.datasets}:{},r=async n=>{if(!n)return n;if("layer"in n){const e=await Promise.all(n.layer.map(r));n={...n,layer:e}}if("hconcat"in n){const e=await Promise.all(n.hconcat.map(r));n={...n,hconcat:e}}if("vconcat"in n){const e=await Promise.all(n.vconcat.map(r));n={...n,vconcat:e}}if(!n.data)return n;if(!("url"in n.data))return n;let i;try{i=e(n.data.url)}catch{return n}const o=await t(i.href,n.data.format);return a[i.pathname]=o,{...n,data:{name:i.pathname}}},i=await r(n);return 0===Object.keys(a).length?i:{...i,datasets:a}}(m),f=[m],o[0]=m,o[1]=p,o[2]=f):(p=o[1],f=o[2]);const{data:h,error:g}=a(p,f);if(g){let e;return o[3]!==g?(e=r.jsx(d,{error:g}),o[3]=g,o[4]=e):e=o[4],e}if(!h)return null;let y;return o[5]!==s||o[6]!==c||o[7]!==l||o[8]!==u||o[9]!==h?(y=r.jsx(K,{value:s,setValue:c,chartSelection:l,fieldSelection:u,spec:h}),o[5]=s,o[6]=c,o[7]=l,o[8]=u,o[9]=h,o[10]=y):y=o[10],y},K=({value:t,setValue:n,chartSelection:a,fieldSelection:d,spec:v})=>{const{theme:k}=i(),x=o.useRef(),[j,b]=o.useState(),_=s(v),P=o.useMemo((()=>O(function(t){return t.data&&"url"in t.data&&(t.data.url=e(t.data.url).href),t}(_),{chartSelection:a,fieldSelection:d})),[_,a,d]),N=o.useMemo((()=>M(P)),[P]),V=c((e=>{n({...t,...e})})),z=s(N),A=o.useMemo((()=>z.reduce(((e,t)=>(S.PAN_ZOOM===t||(e[t]=l(((e,t)=>{f.debug("[Vega signal]",e,t);let n=h.mapValues(t,I);n=h.mapValues(n,Z),V({[e]:n})}),100)),e)),{})),[z,V]),K=c((e=>{f.error(e),f.debug(P),b(e)})),L=c((e=>{f.debug("[Vega view] created",e),x.current=e,b(void 0)}));return r.jsxs(r.Fragment,{children:[j&&r.jsxs(u,{variant:"destructive",children:[r.jsx(m,{children:j.message}),r.jsx("div",{className:"text-md",children:j.stack})]}),r.jsxs("div",{className:"relative",onPointerDown:p.stopPropagation(),children:[r.jsx(w,{spec:P,theme:"dark"===k?"dark":void 0,actions:T,signalListeners:A,onError:K,onNewView:L}),(()=>{const e=[];return S.hasPoint(N)&&e.push(["Point selection","click to select a point; hold shift for multi-select"]),S.hasInterval(N)&&e.push(["Interval selection","click and drag to select an interval"]),S.hasLegend(N)&&e.push(["Legend selection","click to select a legend item; hold shift for multi-select"]),S.hasPanZoom(N)&&e.push(["Pan","hold the meta key and drag"],["Zoom","hold the meta key and scroll"]),0===e.length?null:r.jsx(g,{delayDuration:300,side:"left",content:r.jsx("div",{className:"text-xs flex flex-col",children:e.map(((e,t)=>r.jsxs("div",{children:[r.jsxs("span",{className:"font-bold tracking-wide",children:[e[0],":"]})," ",e[1]]},t)))}),children:r.jsx(y,{className:"absolute bottom-1 right-0 m-2 h-4 w-4 cursor-help text-muted-foreground hover:text-foreground"})})})()]})]})},T={source:!1,compiled:!1};function Z(e){return e instanceof Set?[...e]:e}function I(e){return Array.isArray(e)?e.map((e=>e instanceof Date&&v(e)?new Date(e).getTime():e)):e}export{A as default};
