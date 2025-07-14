const languages = [ 'python', 'javascript', 'cpp' ];
console.log( languages );

for ( const lang of languages ) {
  const script = document.createElement( 'script' );
  script.src = `https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-${ lang }.min.js`;
  document.body.appendChild( script );
  console.log( `Enabled Prism for ${ lang }` );
}
