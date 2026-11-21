// Run the actual embedded page script against a small media/DOM harness.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(process.argv[2],'utf8'),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const buttons=Object.fromEntries(['.play-pair','.pause-pair','.reset-pair'].map(k=>[k,{}]));
const clips=Array.from({length:3},()=>({currentTime:0,paused:true,events:{},play(){this.paused=false;return Promise.resolve()},pause(){this.paused=true},addEventListener(k,f){(this.events[k]??=[]).push(f)},fire(k){(this.events[k]||[]).forEach(f=>f())}}));
const seek={value:0,max:10},out={},search={},visible={};
const card={hidden:false,dataset:{search:'action_01 ue:123 Change action'},querySelectorAll:q=>q==='video.clip'||q==='video'?clips:[],querySelector:q=>q==='.seek'?seek:q==='output'?out:buttons[q]};
const section={hidden:false,querySelectorAll:()=>[card]};
const document={querySelectorAll:q=>q==='article'?[card]:q==='section'?[section]:[],querySelector:q=>q==='#search'?search:visible};
vm.runInNewContext(script,{document});
buttons['.play-pair'].onclick();assert(clips.every(v=>!v.paused));
seek.value=3;seek.oninput();assert(clips.every(v=>v.paused&&v.currentTime===3));assert.equal(out.textContent,'3.00 s');
buttons['.play-pair'].onclick();clips[0].currentTime=5;clips[0].fire('timeupdate');assert.equal(clips[1].currentTime,5);
clips[0].currentTime=10.05;clips[0].fire('timeupdate');assert(clips.every(v=>v.paused&&v.currentTime===10));
buttons['.reset-pair'].onclick();assert(clips.every(v=>v.currentTime===0));
search.oninput({target:{value:'missing'}});assert(card.hidden&&section.hidden);assert.equal(visible.textContent,'0 / 300');
search.oninput({target:{value:'action_01'}});assert(!card.hidden&&!section.hidden);assert.equal(visible.textContent,'1 / 300');
console.log('Review controls passed: play/pause, shared seek, drift correction, endpoint clamp, reset and filter.');
