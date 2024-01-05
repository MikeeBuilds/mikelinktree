const RenderUtils = {
    shaders: {
      vertex: `
  attribute vec3 aVertexPosition;
  
  uniform mat4 uMVMatrix;
  uniform mat4 uPMatrix;
  
  void main(void) {
  gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
  }
      `,
      noise: `
  //
  // Description : Array and textureless GLSL 2D/3D/4D simplex 
  //               noise functions.
  //      Author : Ian McEwan, Ashima Arts.
  //  Maintainer : stegu
  //     Lastmod : 20110822 (ijm)
  //     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
  //               Distributed under the MIT License. See LICENSE file.
  //               https://github.com/ashima/webgl-noise
  //               https://github.com/stegu/webgl-noise
  // 
  precision mediump float;
  
  uniform vec3 iResolution;
  uniform float iGlobalTime;
  uniform float fScale;
  uniform float fSpeed;
  uniform vec2 fOffset;
  
  vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
  }
  
  vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
  }
  
  vec4 permute(vec4 x) {
       return mod289(((x*34.0)+1.0)*x);
  }
  
  vec4 taylorInvSqrt(vec4 r)
  {
    return 1.79284291400159 - 0.85373472095314 * r;
  }
  
  float snoise(vec3 v)
  { 
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
  
  // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;
  
  // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );
  
    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y
  
  // Permutations
    i = mod289(i); 
    vec4 p = permute( permute( permute( 
               i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
             + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
  
  // Gradients: 7x7 points over a square, mapped onto an octahedron.
  // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;
  
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)
  
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
  
    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
  
    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );
  
    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
  
    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
  
    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);
  
  //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
  
  // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                  dot(p2,x2), dot(p3,x3) ) );
  }
  
  #define OCTAVES 4
  float fbm (vec3 st) {
      // Initial values
      float value = 0.0;
      float amplitud = .75;
      float frequency = 0.;
      //
      // Loop of octaves
      for (int i = 0; i < OCTAVES; i++) {
          value += amplitud * snoise(st);
          st *= 2.;
          amplitud *= .5;
      }
      return value;
  }
  
  void main()
  {
      vec2 uv = gl_FragCoord.xy;
    float n = fbm(vec3(fScale * uv + fOffset, iGlobalTime * fSpeed));
    n = (0.5 + 0.5 * n);
    gl_FragColor = vec4(vec3(n),1.0);
  }
      `,
      edge: `
  // Basic sobel filter implementation
  // Jeroen Baert - jeroen.baert@cs.kuleuven.be
  // 
  // www.forceflow.be
  
  
  // Use these parameters to fiddle with settings
  precision mediump float;
  
  uniform float iGlobalTime;
  uniform vec3 iResolution;
  uniform sampler2D iChannel0;
  uniform float fCoarseness;
  
  float quantize(in float fQuantize, in float x) {
    return floor(fQuantize * x) / fQuantize;
  }
  
  float intensity(in float fQuantize, in vec4 color){
    float x = quantize(fQuantize, color.x);
      return sqrt(3.*(x*x));
  }
  
  vec3 sobel(float fQuantize, float stepx, float stepy, vec2 center){
      // get samples around pixel
      float tleft = intensity(fQuantize, texture2D(iChannel0,center + vec2(-stepx,stepy)));
      float left = intensity(fQuantize, texture2D(iChannel0,center + vec2(-stepx,0)));
      float bleft = intensity(fQuantize, texture2D(iChannel0,center + vec2(-stepx,-stepy)));
      float top = intensity(fQuantize, texture2D(iChannel0,center + vec2(0,stepy)));
      float bottom = intensity(fQuantize, texture2D(iChannel0,center + vec2(0,-stepy)));
      float tright = intensity(fQuantize, texture2D(iChannel0,center + vec2(stepx,stepy)));
      float right = intensity(fQuantize, texture2D(iChannel0,center + vec2(stepx,0)));
      float bright = intensity(fQuantize, texture2D(iChannel0,center + vec2(stepx,-stepy)));
   
      // Sobel masks (see http://en.wikipedia.org/wiki/Sobel_operator)
      //        1 0 -1     -1 -2 -1
      //    X = 2 0 -2  Y = 0  0  0
      //        1 0 -1      1  2  1
      
      // You could also use Scharr operator:
      //        3 0 -3        3 10   3
      //    X = 10 0 -10  Y = 0  0   0
      //        3 0 -3        -3 -10 -3
   
      float x = tleft + 2.0*left + bleft - tright - 2.0*right - bright;
      float y = -tleft - 2.0*top - tright + bleft + 2.0 * bottom + bright;
      float color = sqrt((x*x) + (y*y));
      return vec3(color,color,color);
  }
  
  float ring(vec2 uv, vec2 origin, float width, float time, float duration, float delay) {
    // Compute a radially-growing "highlight" circle
    float t = mod(time, duration) / (duration);
    float dist = distance(uv, origin);
    return smoothstep(t, t + 1.0 * width, dist) * smoothstep(t + 3.0 * width, t + 2.0 * width, dist);
  }
  
  void main(){
      vec2 uv = gl_FragCoord.xy / iResolution.xy;
  
    // Take a low-contrast sample of the original noise texture
    vec3 tex = vec3(0.25 * texture2D(iChannel0, uv));
  
    // Make a highlight with several rings
    float ring1 = ring(uv, vec2(0.5), 0.01, iGlobalTime, 15.0, 30.0);
    // float ring2 = ring(uv, vec2(0.25), 0.05, iGlobalTime + 22.0, 13.0, 3.0);
    // float ring3 = ring(uv, vec2(0.75, 0.33), 0.005, iGlobalTime + 935.0, 27.0, 50.0);
    float ring4 = ring(uv, vec2(0.5), 0.01, iGlobalTime + 3.0, 19.0, 10.0);
    // vec3 highlight = vec3(0.0, ring1 + 0.5 * ring2 + ring3 + ring4, 0.0, 12.0);
    vec3 highlight = vec3(0.0, ring1 + ring4, 0.0);
  
    // Blend highlight color with detected edges
    vec3 firstPass = clamp((2.0 * sobel(2.0, 1.0/iResolution[0], 1.0/iResolution[1], uv)), 0.0, 1.0);
    vec3 secondPass = clamp((1.0 * sobel(16.0, 1.5/iResolution[0], 1.5/iResolution[1], uv)), 0.0, 1.0);
    vec3 passes = firstPass + secondPass;
    vec3 lines = 0.75 * passes * highlight + 0.25 * passes;
  
    // Screen blend mode
      gl_FragColor = vec4(1.0 - (1.0 - tex) * (1.0 - lines), 1.0);
  }
      `
    },
    render(canvas, updateUniformsCallback) {
      const { width, height } = canvas;
      
      // Init WebGL context
      const gl = canvas.getContext('webgl');
      gl.clearColor(0, 0, 0, 1);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      
      // Init noise shader
      const shaders = {};
      const vertexShader = gl.createShader(gl.VERTEX_SHADER);
      gl.shaderSource(vertexShader, this.shaders.vertex);
      gl.compileShader(vertexShader);
      if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
        console.error('Unable to compile vertex shader: ' + gl.getShaderInfoLog(vertexShader));
        return;
      }
  
      const noiseShader = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(noiseShader, this.shaders.noise);
      gl.compileShader(noiseShader);
      if (!gl.getShaderParameter(noiseShader, gl.COMPILE_STATUS)) {
        console.error('Unable to compile fragment shader: ' + gl.getShaderInfoLog(noiseShader));
        return;
      }
      
      const noiseProgram = gl.createProgram();
      gl.attachShader(noiseProgram, vertexShader);
      gl.attachShader(noiseProgram, noiseShader);
      gl.linkProgram(noiseProgram);
      if (!gl.getProgramParameter(noiseProgram, gl.LINK_STATUS)) {
        console.error('Unable to initialize shader program: ' + gl.getProgramInfoLog(noiseProgram));
        return;
      }
      shaders.noise = {
        program: noiseProgram
      };
      
      // Init edge shader
      const edgeShader = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(edgeShader, this.shaders.edge);
      gl.compileShader(edgeShader);
      if (!gl.getShaderParameter(edgeShader, gl.COMPILE_STATUS)) {
        console.error('Unable to compile edge fragment shader: ' + gl.getShaderInfoLog(edgeShader));
        return;
      }
      
      const edgeProgram = gl.createProgram();
      gl.attachShader(edgeProgram, vertexShader);
      gl.attachShader(edgeProgram, edgeShader);
      gl.linkProgram(edgeProgram);
      if (!gl.getProgramParameter(edgeProgram, gl.LINK_STATUS)) {
        console.error('Unable to initialize edge shader program: ' + gl.getProgramInfoLog(edgeProgram));
        return;
      }
      shaders.edge = {
        program: edgeProgram,
      };
      
      // Set up attributes and uniforms
      gl.useProgram(noiseProgram);
      
      shaders.noise.uniforms = {};
      shaders.noise.uniforms.resolution = gl.getUniformLocation(noiseProgram, 'iResolution');
      shaders.noise.uniforms.time = gl.getUniformLocation(noiseProgram, 'iGlobalTime');
      shaders.noise.uniforms.scale = gl.getUniformLocation(noiseProgram, 'fScale');
      shaders.noise.uniforms.speed = gl.getUniformLocation(noiseProgram, 'fSpeed');
      shaders.noise.uniforms.offset = gl.getUniformLocation(noiseProgram, 'fOffset');
      shaders.noise.uniforms.perspective = gl.getUniformLocation(noiseProgram, "uPMatrix");
      shaders.noise.uniforms.modelview = gl.getUniformLocation(noiseProgram, "uMVMatrix");
  
      shaders.noise.attributes = {};
      shaders.noise.attributes.vertexPos = gl.getAttribLocation(noiseProgram, 'aVertexPosition');
      
      gl.useProgram(edgeProgram);
  
      shaders.edge.uniforms = {};
      shaders.edge.uniforms.resolution = gl.getUniformLocation(edgeProgram, 'iResolution');
      shaders.edge.uniforms.time = gl.getUniformLocation(edgeProgram, 'iGlobalTime');
      shaders.edge.uniforms.texture = gl.getUniformLocation(edgeProgram, 'iChannel0');
      shaders.edge.uniforms.coarseness = gl.getUniformLocation(edgeProgram, 'fCoarseness');
      // shaders.edge.uniforms.quantize = gl.getUniformLocation(edgeProgram, 'fQuantize');
      shaders.edge.uniforms.perspective = gl.getUniformLocation(edgeProgram, "uPMatrix");
      shaders.edge.uniforms.modelview = gl.getUniformLocation(edgeProgram, "uMVMatrix");
      
      shaders.edge.attributes = {};
      shaders.edge.attributes.vertexPos = gl.getAttribLocation(edgeProgram, 'aVertexPosition');    
      
      // Init framebuffer
      const framebuffer = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      
      // Init buffer
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      
      const vertices = [
         1.0,  1.0,  0.0,
        -1.0,  1.0,  0.0,
         1.0, -1.0,  0.0,
        -1.0, -1.0,  0.0,
      ];
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      
      // Set up camera
      const perspective = mat4.ortho(mat4.create(), -1, 1, -1, 1, 0, 1);
      const mvMatrix = mat4.fromTranslation(mat4.create(), [0, 0, -1]);
      let uniforms = updateUniformsCallback();
      const loop = () => {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        
        // Attach to framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.viewport(0, 0, width, height);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        
        // Switch to noise texture
        gl.useProgram(noiseProgram);
        gl.enableVertexAttribArray(shaders.noise.attributes.vertexPos);
        gl.vertexAttribPointer(shaders.noise.attributes.vertexPos, 3, gl.FLOAT, false, 0, 0);
  
        gl.uniform3fv(shaders.noise.uniforms.resolution, [width, height, 1]);
        gl.uniform1f(shaders.noise.uniforms.time, uniforms.time);
        gl.uniform1f(shaders.noise.uniforms.scale, uniforms.scale);
        gl.uniform1f(shaders.noise.uniforms.speed, uniforms.speed);
        gl.uniform2fv(shaders.noise.uniforms.offset, uniforms.offset);
        gl.uniformMatrix4fv(shaders.noise.uniforms.perspective, false, new Float32Array(perspective));
        gl.uniformMatrix4fv(shaders.noise.uniforms.modelview, false, new Float32Array(mvMatrix));
  
        // Draw texture
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        
        // Switch to edge shader
        gl.useProgram(edgeProgram);
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.enableVertexAttribArray(shaders.edge.attributes.vertexPos);
        gl.vertexAttribPointer(shaders.edge.attributes.vertexPos, 3, gl.FLOAT, false, 0, 0);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        
        gl.uniform3fv(shaders.edge.uniforms.resolution, [width, height, 1]);
        gl.uniform1i(shaders.edge.uniforms.texture, 0);
        gl.uniform1f(shaders.edge.uniforms.time, uniforms.time);
        gl.uniform1f(shaders.edge.uniforms.coarseness, uniforms.coarseness);
        gl.uniformMatrix4fv(shaders.edge.uniforms.perspective, false, new Float32Array(perspective));
        gl.uniformMatrix4fv(shaders.edge.uniforms.modelview, false, new Float32Array(mvMatrix));
  
        // Render to screen
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        
        // Prepare for next render
        uniforms = updateUniformsCallback();
        
        requestAnimationFrame(loop);
      };
      loop();
    }
  };
  
  Vue.component('edge-graphic', {
    template: `
  <div>
    <div class="graphic-container" :style="{ width: width + 'px', height: height + 'px' }">
      <canvas :width="width" :height="height" :style="{ width: width + 'px', height: height + 'px' }"></canvas>
      <svg :width="width" :height="height">
        <path d="M 50,5 95,97.5 5,97.5 z" :transform="transform(p)" fill="red" fill-opacity="0.8" v-for="p in points" />
      </svg>
    </div>
  
    <div>
      <label>Step</label>
      <input type="text" :value="uniforms.time | round" readonly>
    </div>
    <div>
      <label>Scale</label>
      <input type="range" v-model="uniforms.scale" min="0.001" max="0.01" step="0.0005">
      <span>{{ uniforms.scale }}</span>
    </div>
    <div>
      <label>Speed</label>
      <input type="range" v-model="uniforms.speed" min="0.0001" max="0.01" step="0.00005">
      <span>{{ uniforms.speed }}</span>
    </div>
    <div>
      <label>Line coarseness</label>
      <input type="range" v-model="uniforms.coarseness" min="0.25" max="2.5" step="0.05">
      <span>{{ uniforms.coarseness }}</span>
    </div>
    <div>
      <label>Quantization</label>
      <input type="range" v-model="uniforms.quantize" min="2" max="32" step="2">
      <span>{{ uniforms.quantize }}</span>
    </div>
  </div>`,
    props: [
      'width',
      'height',
    ],
    data() {
      return {
        points: [
          { x: 0.1 * this.width, y: 0.1 * this.height, dx: 1, dy: -1 },
          { x: 0.5 * this.width, y: 0.25 * this.height, dx: 0.5, dy: 2 },
          { x: 0.9 * this.width, y: 0.4 * this.height, dx: 0.25, dy: -1 },
          { x: 0.4 * this.width, y: 0.9 * this.height, dx: -0.3, dy: 0.5 },
          { x: 0.6 * this.width, y: 0.5 * this.height, dx: 0, dy: 0 },
        ],
        uniforms: {
          time: 0,
          scale: 1.0 / 500.0,
          speed: 1.0 / 1000.0,
          coarseness: 0.5,
          offset: [
            1000 * Math.random(),
            1000 * Math.random()
          ],
          quantize: 16
        }
      };
    },
    filters: {
      round(f) {
        return Math.floor(f * 10) / 10;
      }
    },
    mounted() {
      this.render();
    },
    methods: {
      render() {
        this.t = window.performance.now();
        RenderUtils.render(this.$el.querySelector('canvas'), this.tick.bind(this));
      },
      transform(p) {
        return `translate(${p.x},${p.y}) scale(0.1)`;
      },
      tick() {
        this.uniforms.time += 0.1;
        this.points.forEach(p => {
          p.x += 0.1 * p.dx;
          p.y += 0.1 * p.dy;
          
          if (p.x > this.width) p.x -= this.width;
          if (p.x < 0) p.x += this.width;
          if (p.y > this.height) p.y -= this.height;
          if (p.y < 0) p.y += this.height;
        })
        
        return this.uniforms;
      }
    }
  });
  
  new Vue({
    el: '#app',
    data() {
      return {
        options: {
          width: window.innerWidth,
          height: window.innerWidth,
          // height: window.innerHeight,
        }
      };
    }
  });