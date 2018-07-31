<template>
  <div id="app">
    <img src="./assets/logo.png">
    <Annotation text="titledasdaddasd" v-bind:start="2" v-bind:end="6" idelem="1" ent="LOC"></Annotation>
  </div>
</template>

<script>
import Annotation from './components/Annotation.vue'

export default {
  name: 'app',
  components: {
    Annotation
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
