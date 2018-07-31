<template>
  <div>
    <p>{{ firstPartText }}<mark v-bind:data-idelem="idelem" v-bind:data-ent="ent">{{ annotationPartText }}</mark>{{ secondPartText }}</p>
  </div>

</template>
<script>
export default {
  name: 'Annotation',
  props: {
    idelem: String,
    text: String,
    start: Number,
    end: Number,
    ent: String
    },
  computed: {
    firstPartText: function () {
      return this.text.slice(0, this.start)
    },
    secondPartText: function () {
      return this.text.slice(this.end)
    },
    annotationPartText: function () { 
      return this.text.slice(this.start, this.end)
    }
  }
}
</script>

<style scoped>

mark {
  -moz-border-radius: 6px;
 border-radius: 6px;
}

[data-ent] {
    line-height: 1.5em;
    display: inline-block;
    padding: 0 0.15em 0.15em 2.5em;
    position: relative;
    cursor: help;
    white-space: pre;
}

[data-ent]:after {
    box-sizing: border-box;
    content: '*' attr(data-ent);
    line-height: 1;
    display: inline-block;
    font-size: 0.65em;
    font-weight: bold;
    text-transform: uppercase;
    position: absolute;
    /*left: 2em;*/
    left: 0.5em;
    top: 0.75em;
    -webkit-transition: opacity 0.25s ease;
    transition: opacity 0.25s ease;
}

[data-ent][data-ent="LOC"] {
  background-color: rgba(41, 122, 196, 0.80);
}

[data-ent][data-ent="PER"] {
  background-color: rgba(166, 54, 35, 0.8);
}

[data-ent][data-ent="ORG"] {
  background-color: rgba(14, 185, 36, 0.8);
}

[data-ent][data-ent="MISC"] {
  background-color: rgba(207, 198, 74, 0.8);
}
</style>
